import torch
import torch.nn as nn
import torchvision.models as models


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=upsample_scale, stride=upsample_scale),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ResNetAutoencoder(nn.Module):
    def __init__(self, use_pretrained: bool):
        super().__init__()

        self.cnn_version = "res_net"

        resnet_kwargs = {
            "weights": "IMAGENET1K_V1"} if use_pretrained else {}
        self.res_net = models.resnet50(**resnet_kwargs)
        if use_pretrained:
            for param in self.res_net.parameters():
                param.requires_grad = False
            for param in self.res_net.layer4.parameters():
                param.requires_grad = True

        self.encoder = nn.Sequential(*list(self.res_net.children())[:-2])

        self.decoder = nn.Sequential(
            # Corresponds to the last block of ResNet-50
            UpSampleBlock(2048, 1024),
            UpSampleBlock(1024, 512),
            UpSampleBlock(512, 256),
            UpSampleBlock(256, 128),
            UpSampleBlock(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Original ResNet starts with a 7x7 conv
            nn.Conv2d(64, 3, kernel_size=7, stride=2, padding=3),
            nn.Sigmoid()  # input images are normalized to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    ae = ResNetAutoencoder(use_pretrained=True)
    inp = torch.randn((8, 3, 224, 224))
    out = ae(inp)
    print(f"{out.shape=}")
    loss = torch.nn.functional.mse_loss(inp, out)
    print(f"{loss=}")
    loss.backward()
