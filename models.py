import torch
import torch.nn as nn

from torchvision import models


class VisionRobotNet(nn.Module):
    def __init__(self,
                 num_image_features: int,
                 num_robot_features: int,
                 dropout_rate: float) -> None:
        super().__init__()
        self.res_net_left = self._init_res_net(num_image_features)
        self.res_net_right = self._init_res_net(num_image_features)

        self.num_image_features = num_image_features
        self.num_robot_features = num_robot_features

        self.fc1 = nn.Linear(2 * num_image_features + num_robot_features, 84)
        self.fc2 = nn.Linear(84, 180)
        self.fc3 = nn.Linear(180, 50)
        self.fc4 = nn.Linear(50, 3)

        self.bn1 = nn.BatchNorm1d(84)
        self.bn2 = nn.BatchNorm1d(180)
        self.bn3 = nn.BatchNorm1d(50)

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

    def forward(self, img_right: torch.Tensor, img_left: torch.Tensor, x: torch.Tensor):
        img_right_features = self.res_net_right(img_right)
        img_left_features = self.res_net_left(img_left)

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
