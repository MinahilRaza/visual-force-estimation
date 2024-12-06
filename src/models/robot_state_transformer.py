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

        self.transformer = nn.Transformer(
            d_model=config.hidden_layers[-1],
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers, # might not be needed
            dim_feedforward=config.dim_feedforward, # 4 times attention dimension D (encoder output)
            dropout=config.dropout_rate, # 0.1, 0.2
            norm_first = False # True
        )

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

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        """
        robot_state: [batch_size, seq_length, num_robot_features]
        """
        batch_size, seq_length, _ = robot_state.size()

        # Flatten to apply 1D DenseNet encoder
        if self.encoder_state == EncoderState.LINEAR:
            robot_state = robot_state.view(-1, self.num_robot_features)
        encoded_features = self.robot_encoder(robot_state)
        if self.encoder_state == EncoderState.LINEAR:
            encoded_features = encoded_features.view(
                batch_size, seq_length, -1)

        # Transformer expects inputs in (seq_length, batch_size, features)
        if self.encoder_state == EncoderState.LINEAR:
            encoded_features = encoded_features.permute(1, 0, 2)
        transformer_output = self.transformer(
            encoded_features, encoded_features)

        # Output layer
        output = self.output_layer(transformer_output)
        if self.encoder_state == EncoderState.LINEAR:
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
