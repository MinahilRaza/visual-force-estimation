import torch
import torch.nn as nn
import torchvision.models as models


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


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
    def __init__(self, base_model: str, use_pretrained: bool):
        super().__init__()

        self.cnn_version = base_model

        if base_model == 'resnet50':
            resnet_model = models.resnet50
        elif base_model == 'resnet18':
            resnet_model = models.resnet18
        else:
            raise ValueError("Unsupported base model type")

        resnet_kwargs = {"weights": "IMAGENET1K_V1"} if use_pretrained else {}
        self.res_net = resnet_model(**resnet_kwargs)

        if use_pretrained:
            for param in self.res_net.parameters():
                param.requires_grad = False
            for param in self.res_net.layer4.parameters():
                param.requires_grad = True

        self.encoder = nn.Sequential(*list(self.res_net.children())[:-2])

        if base_model == 'resnet50':
            decoder_blocks = [UpSampleBlock(2048, 1024),
                              UpSampleBlock(1024, 512),
                              UpSampleBlock(512, 256),
                              UpSampleBlock(256, 128),
                              UpSampleBlock(128, 64),
                              #   nn.ConvTranspose2d(
                              #       64, 64, kernel_size=2, stride=2),
                              #   nn.BatchNorm2d(64),
                              #   nn.ReLU(inplace=True),
                              # Original ResNet should start with a 7x7 conv
                              nn.Conv2d(64, 3, kernel_size=3,
                                        stride=1, padding=1),
                              nn.Sigmoid()]
        else:
            decoder_blocks = [UpSampleBlock(512, 256),
                              UpSampleBlock(256, 128),
                              UpSampleBlock(128, 64),
                              UpSampleBlock(64, 32),
                              UpSampleBlock(32, 16),
                              nn.ConvTranspose2d(
                                  16, 16, kernel_size=2, stride=2),
                              nn.BatchNorm2d(16),
                              nn.ReLU(inplace=True),
                              # Original ResNet starts with a 7x7 conv
                              nn.Conv2d(16, 3, kernel_size=7,
                                        stride=2, padding=3),
                              nn.Sigmoid()]

        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    ae = ResNetAutoencoder(base_model="resnet18", use_pretrained=True)
    inp = torch.randn((4, 3, 224, 224))
    out = ae(inp)
    assert not torch.isnan(out).any()
    loss = torch.nn.functional.mse_loss(inp, out)
    print(f"{loss=}")
    loss.backward()
