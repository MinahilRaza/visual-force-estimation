import torch
import torch.nn as nn

from torchvision import models


class VisionRobotNet(nn.Module):
    def __init__(self,
                 cnn_model_version: str,
                 num_image_features: int,
                 num_robot_features: int,
                 dropout_rate: float = 0.2) -> None:
        super().__init__()
        if cnn_model_version == "res_net":
            self.cnn_left = self._init_res_net(num_image_features)
            self.cnn_right = self._init_res_net(num_image_features)
        else:
            self.cnn_left = self._init_efficient_net(
                num_image_features, version=cnn_model_version)
            self.cnn_right = self._init_efficient_net(
                num_image_features, version=cnn_model_version)

        self.num_image_features = num_image_features
        self.num_robot_features = num_robot_features

        self.fc1 = nn.Linear(2 * num_image_features + num_robot_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    @staticmethod
    def _init_res_net(num_image_features: int) -> models.ResNet:
        res_net = models.resnet50(weights='IMAGENET1K_V1')
        num_res_net_features = res_net.fc.in_features

        for p in res_net.parameters():
            p.requires_grad = False

        res_net.fc = nn.Linear(
            num_res_net_features, num_image_features)
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

    def forward(self, img_right: torch.Tensor, img_left: torch.Tensor, x: torch.Tensor):
        img_right_features = self.cnn_right(img_right)
        img_left_features = self.cnn_left(img_left)

        x = torch.cat((img_left_features, img_right_features, x), dim=-1)

        x = self._linear_forward(x, 1)
        x = self._linear_forward(x, 2)
        x = self._linear_forward(x, 3)

        out = self.fc4(x)
        return out

    def _linear_forward(self, x: torch.Tensor, layer_nr: int):
        linear_layer = getattr(self, f"fc{layer_nr}")
        batch_norm_layer = getattr(self, f"bn{layer_nr}")

        x = linear_layer(x)
        x = batch_norm_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    @property
    def device(self) -> torch.device:
        return self.fc1.weight.device


if __name__ == "__main__":
    img_r = torch.randn((8, 3, 256, 256))
    img_l = torch.randn((8, 3, 256, 256))
    feat = torch.randn((8, 41))

    model = VisionRobotNet("efficientnet_v2_m", 30, 41, dropout_rate=0.2)
    out = model(img_r, img_l, feat)

    assert not torch.isnan(out).any()
