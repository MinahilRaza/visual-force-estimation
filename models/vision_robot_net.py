import os
import torch
import torch.nn as nn

from torchvision import models
from models.var_auto_encoder import ResNet50Enc

from dataclasses import dataclass

import constants


@dataclass
class VRNConfig:
    cnn_model_version: str
    num_image_features: int
    num_robot_features: int
    hidden_layers: list
    use_pretrained: bool
    dropout_rate: float
    use_batch_norm: bool


class VisionRobotNet(nn.Module):
    def __init__(self, config: VRNConfig) -> None:
        super().__init__()
        self.cnn_version = config.cnn_model_version
        if config.cnn_model_version == "res_net":
            self.cnn = self._init_res_net(
                config.num_image_features, config.use_pretrained)
        elif config.cnn_model_version.startswith("efficientnet"):
            self.cnn = self._init_efficient_net(
                config.num_image_features, version=config.cnn_model_version)
        else:
            self.cnn = self._init_finetuned_res_net(
                config.num_image_features, config.cnn_model_version)

        self.num_image_features = config.num_image_features
        self.num_robot_features = config.num_robot_features

        self.dropout_rate = config.dropout_rate
        self.use_batch_norm = config.use_batch_norm
        self.config = config

        self._initialize_linear_layers(config)
        self._initialize_weights()

    def _initialize_linear_layers(self, config: VRNConfig) -> None:
        layers = []
        in_features = 2 * config.num_image_features + config.num_robot_features
        for out_features in config.hidden_layers:
            layers.append(self._make_linear_layer(in_features, out_features))
            in_features = out_features
        self.linear_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_features, 3)

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

    @staticmethod
    def _init_res_net(num_image_features: int, use_pretrained: bool) -> models.ResNet:
        res_net_kwargs = {"weights": "IMAGENET1K_V1"} if use_pretrained else {}
        res_net = models.resnet50(**res_net_kwargs)
        num_res_net_features = res_net.fc.in_features

        if use_pretrained:
            for p in res_net.parameters():
                p.requires_grad = False

        res_net.fc = nn.Linear(
            num_res_net_features, num_image_features)
        return res_net

    @staticmethod
    def _init_finetuned_res_net(num_image_features: int, weights_dir: str) -> models.ResNet:
        assert os.path.isdir(weights_dir), f"{weights_dir=}"

        enc_model = ResNet50Enc(enc_dim=num_image_features)
        weights_path = os.path.join(weights_dir, constants.ENCODER_WEIGHTS_FN)

        state_dict = torch.load(weights_path)
        enc_model.load_state_dict(state_dict)

        res_net = enc_model.res_net
        for p in res_net.parameters():
            p.requires_grad = False

        num_features = res_net.fc.in_features
        res_net.fc = nn.Linear(num_features, num_image_features)

        return res_net

    @staticmethod
    def _init_efficient_net(num_image_features: int, version: str) -> models.EfficientNet:
        if version == "efficientnet_v2_m":
            efficient_net = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        elif version == "efficientnet_b0":
            efficient_net = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif version == "efficientnet_b1":
            efficient_net = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid EfficientNet Version: {version}")

        for param in efficient_net.parameters():
            param.requires_grad = False

        num_features = efficient_net.classifier[1].in_features
        efficient_net.classifier[1] = nn.Linear(
            num_features, num_image_features)

        return efficient_net

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img_right: torch.Tensor, img_left: torch.Tensor, x: torch.Tensor):
        img_right_features = self.cnn(img_right)
        img_left_features = self.cnn(img_left)
        x = torch.cat((img_left_features, img_right_features, x), dim=-1)

        for layer in self.linear_layers:
            x = layer(x)
        out = self.output_layer(x)

        return out

    @property
    def device(self) -> torch.device:
        return self.fc1.weight.device


if __name__ == "__main__":
    img_r = torch.randn((8, 3, 256, 256))
    img_l = torch.randn((8, 3, 256, 256))
    feat = torch.randn((8, 41))

    basic_config = VRNConfig("efficientnet_v2_m", 30, 41, hidden_layers=[128, 512, 64],
                             use_pretrained=False, dropout_rate=0.2, use_batch_norm=False)
    model = VisionRobotNet(basic_config)
    out = model(img_r, img_l, feat)

    assert not torch.isnan(out).any()
