# https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3,
                             scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet50Enc(nn.Module):
    def __init__(self, enc_dim: int) -> None:
        super().__init__()
        self.enc_dim = enc_dim

        self.res_net = models.resnet50(weights="IMAGENET1K_V1")
        num_res_net_features = self.res_net.fc.in_features
        self.res_net.fc = nn.Linear(num_res_net_features, 2 * enc_dim)
        for param in self.res_net.conv1.parameters():
            param.requires_grad = False
        for param in self.res_net.bn1.parameters():
            param.requires_grad = False
        for param in self.res_net.layer1.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res_net(x)

        # Split the output into mu and logvar
        mu = x[:, :self.enc_dim]
        logvar = x[:, self.enc_dim:]

        return mu, logvar


class ResNet50Dec(nn.Module):
    def __init__(self, num_Blocks=[3, 4, 6, 3], enc_dim=10, nc=3):
        super(ResNet50Dec, self).__init__()
        self.in_planes = 2048
        # Linear layer to map z_dim to a spatial size that matches the start of decoding
        self.linear = nn.Linear(enc_dim, self.in_planes * 7 * 7)

        # Reverse the blocks relative to ResNet50 structure
        self.layer4 = self._make_layer(1024, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(512, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(256, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(128, num_Blocks[0], stride=2)

        # Final convolution to get to the desired number of channels
        self.conv1 = ResizeConv2d(128, nc, kernel_size=3, scale_factor=2)
        self.in_planes = 2048

    def _make_layer(self, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), self.in_planes, 7, 7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        # Final resize and convolution to get to the output size and channel number
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x


class VarAutoEncoder(nn.Module):

    def __init__(self, enc_dim: int):
        super().__init__()
        self.encoder = ResNet50Enc(enc_dim)
        self.decoder = ResNet50Dec(enc_dim=enc_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, z, mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor):
        # in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean


if __name__ == "__main__":
    model = VarAutoEncoder(enc_dim=30)
    inp = torch.rand((8, 3, 224, 224))
    out, z = model(inp)
    print(f"{out.shape=}")
    print(f"{z.shape=}")
