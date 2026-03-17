from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(ch, ch),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResNet9(nn.Module):
    """
    Compact ResNet-9 variant for CIFAR-10.
    Returns a pooled feature vector (B, feat_dim).
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 512):
        super().__init__()
        self.conv1 = _conv_bn_relu(in_channels, 64)
        self.conv2 = _conv_bn_relu(64, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(128)

        self.conv3 = _conv_bn_relu(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.conv4 = _conv_bn_relu(256, feat_dim)
        self.pool3 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(feat_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.res2(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def build_encoder(backbone: str = "resnet9") -> tuple[nn.Module, int]:
    backbone = backbone.lower()
    if backbone == "resnet9":
        enc = ResNet9(in_channels=3, feat_dim=512)
        return enc, 512
    raise ValueError(f"Unknown backbone: {backbone}")

