# TODO: try batch first to get rid of the permute
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from dataclasses import dataclass
from enum import Enum


class EncoderState(Enum):
    LINEAR = 1
    CONV = 2


@dataclass
class TransformerConfig:
    num_robot_features: int
    hidden_layers: List[int]
    dropout_rate: float
    use_batch_norm: bool
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    encoder_state: EncoderState
    max_seq_length: int = 512
    use_positional_encoding: bool = True  # TODO remove true and pass as args
    is_causal: bool = True  # TODO remove true and pass as args


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create positional encoding matrix
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.
        x: [batch_size, seq_length, d_model]
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RobotStateTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, seed: int = 42) -> None:
        super().__init__()
        self.version = "transformer"
        self.num_robot_features = config.num_robot_features
        self.max_seq_length = config.max_seq_length
        self.encoder_state = config.encoder_state
        self.config = config

        if self.encoder_state == EncoderState.LINEAR:
            self.robot_encoder = LinearEncoder(
                num_input_features=self.num_robot_features,
                hidden_layers=config.hidden_layers,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm
            )
        else:
            self.robot_encoder = DenseNetEncoder(
                num_input_features=self.num_robot_features,
                hidden_layers=config.hidden_layers,
                growth_rate=32,
                num_blocks=4,
                dropout_rate=config.dropout_rate
            )

        # Encoder for ground truth output
        self.output_encoder = LinearEncoder(
            # Assuming 3 output features (x, y, z forces)
            num_input_features=3,
            hidden_layers=config.hidden_layers,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm
        ) if self.config.is_causal else None

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_layers[-1],
            max_seq_length=config.max_seq_length,
            dropout_rate=config.dropout_rate
        ) if config.use_positional_encoding else None

        if self.positional_encoding is not None:
            print("Using positional encoding")
        if self.config.is_causal:
            print("Using causal mask")

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_layers[-1],
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout_rate,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # self.transformer = nn.Transformer(
        #     d_model=config.hidden_layers[-1],
        #     nhead=config.num_heads,
        #     num_encoder_layers=config.num_encoder_layers,
        #     num_decoder_layers=config.num_decoder_layers, # might not be needed
        #     dim_feedforward=config.dim_feedforward, # 4 times attention dimension D (encoder output)
        #     dropout=config.dropout_rate, # 0.1, 0.2
        #     norm_first = False # True
        # )

        self.output_layer = nn.Linear(config.hidden_layers[-1], 3)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed: int):
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Generate a custom causal mask where at timestep t:
        - All input features for timesteps 1 to t are available.
        - Only predicted output features from timesteps 1 to t-1 are available.
        This matrix will be added to our target vector, so the matrix will be made
        up of zeros in the positions where the transformer can have access to the 
        elements, and minus infinity where it can’t.
        """
        mask = torch.full((seq_length, seq_length), float('-inf'))
        # Set values on and above the diagonal to 0
        mask = torch.triu(mask, diagonal=0)
        return mask

    def forward(self, robot_state: torch.Tensor, ground_truth: torch.Tensor = None) -> torch.Tensor:
        # TODO make this code compatible with non-causal case
        """
        robot_state: [batch_size, seq_length, num_robot_features]
        ground_truth: [batch_size, seq_length, 3]  # Assuming 3 output features
        """
        batch_size, seq_length, _ = robot_state.size()

        # Encode robot state
        if self.encoder_state == EncoderState.LINEAR:
            robot_state = robot_state.view(-1, self.num_robot_features)
        encoded_robot_features = self.robot_encoder(robot_state)
        if self.encoder_state == EncoderState.LINEAR:
            encoded_robot_features = encoded_robot_features.view(
                batch_size, seq_length, -1)

        # Add positional encoding to robot state
        if self.config.use_positional_encoding:
            encoded_robot_features = self.positional_encoding(
                encoded_robot_features)

        # Encode ground truth output
        if self.config.is_causal:
            ground_truth = ground_truth.view(-1, 3)
            encoded_output_features = self.output_encoder(ground_truth)
            encoded_output_features = encoded_output_features.view(
                batch_size, seq_length, -1)

            # Add positional encoding to force features
            if self.config.use_positional_encoding:
                encoded_output_features = self.positional_encoding(
                    encoded_output_features)

            # Interleave the features [robot_state_encoded_1 (r1), force_encoded_1 (f1),
            #                       robot_state_encoded_2 (r2), force_encoded_2 (f2), ...]

            combined_features = torch.cat(
                [encoded_robot_features, encoded_output_features], dim=-1).view(batch_size, 2 * seq_length, -1)

        else:
            combined_features = encoded_robot_features

        # Transformer expects inputs in (seq_length, batch_size, features)
        combined_features = combined_features.permute(1, 0, 2)

        # Causal Mask
        # t = 1 : [r1, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, ...]
        # t = 2 : [r1,   f1,   r2, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, ...]
        # t = 3 : [r1,   f1,   r2,   f2,   r3, -Inf, -Inf, -Inf, -Inf, -Inf, ...]

        # Apply causal masking if enabled
        if self.config.is_causal:
            mask = self.generate_causal_mask(
                seq_length).to(combined_features.device)
        else:
            mask = None

        transformer_output = self.transformer_encoder(
            combined_features, mask=mask, is_causal=True)

        # Output layer
        output = self.output_layer(transformer_output)
        output = output.permute(1, 0, 2)  # [batch_size, seq_length, 3]

        return output


class LinearEncoder(nn.Module):
    def __init__(self, num_input_features: int, hidden_layers: List[int], dropout_rate: float, use_batch_norm: bool) -> None:
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self._initialize_linear_layers(num_input_features, hidden_layers)

    def _initialize_linear_layers(self, num_input_features: int, hidden_layers: List[int]) -> None:
        layers = []
        in_features = num_input_features
        for out_features in hidden_layers:
            layers.append(self._make_linear_layer(in_features, out_features))
            in_features = out_features
        self.linear_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_features, hidden_layers[-1])

    def _make_linear_layer(self, in_features: int, out_features: int) -> nn.Module:
        if self.use_batch_norm:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linear_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out


class DenseNetEncoder(nn.Module):
    def __init__(self, num_input_features: int, hidden_layers: List[int], growth_rate: int, num_blocks: int, dropout_rate: float) -> None:
        super().__init__()
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        # Initial Convolution
        self.conv1 = nn.Conv1d(
            num_input_features, hidden_layers[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks
        self.dense_blocks = self._make_dense_blocks(hidden_layers)

        # Final BatchNorm
        self.bn2 = nn.BatchNorm1d(hidden_layers[-1])

        # Fully connected layer
        self.fc = nn.Linear(hidden_layers[-1], hidden_layers[-1])

    def _make_dense_blocks(self, hidden_layers: List[int]) -> nn.Sequential:
        layers = []
        in_channels = hidden_layers[0]
        for out_channels in hidden_layers[1:]:
            layers.append(self._make_dense_block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_dense_block(self, in_channels: int, out_channels: int) -> nn.Module:
        layers = [
            nn.Conv1d(in_channels, self.growth_rate,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.growth_rate),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(self.growth_rate, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # Change to [batch_size, features, seq_length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.dense_blocks(x)
        x = self.bn2(x)
        x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)

        out = self.fc(x)
        return out


if __name__ == "__main__":
    # Example usage
    seq_length = 10
    num_robot_features = 58
    batch_size = 8

    config = TransformerConfig(
        num_robot_features=num_robot_features,
        hidden_layers=[128, 256],
        dropout_rate=0.2,
        use_batch_norm=True,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        encoder_state=EncoderState.CONV
    )

    model = RobotStateTransformer(config)
    robot_state = torch.randn((batch_size, seq_length, num_robot_features))
    output = model(robot_state)

    print(output.shape)  # Expected output: [batch_size, seq_length, 3]
