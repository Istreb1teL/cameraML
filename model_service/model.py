import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()

        # Encoder (используем pretrained ResNet)
        backbone = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        # Decoder
        self.upconv4 = DoubleConv(512 + 256, 256)
        self.upconv3 = DoubleConv(256 + 128, 128)
        self.upconv2 = DoubleConv(128 + 64, 64)
        self.upconv1 = DoubleConv(64 + 64, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(e1)  # 64
        e3 = self.encoder3(e2)  # 128
        e4 = self.encoder4(e3)  # 256
        e5 = self.encoder5(e4)  # 512

        # Decoder
        d4 = self.upsample(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.upconv4(d4)

        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.upconv3(d3)

        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.upconv2(d2)

        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.upconv1(d1)

        out = self.final_conv(d1)
        return torch.sigmoid(out)