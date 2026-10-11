# -*- coding: utf-8 -*-
"""
unet.py

U-Net baseline exactly as specified in the MARIDA paper
(Kikaki et al., PLOS ONE 2022, "Machine learning frameworks"):

  * Input layer adapted to the 11 Rayleigh-corrected S2 bands
    (or 11 + extra feature channels when spectral indices/texture
    features are stacked on).
  * 4 down-sampling and 4 up-sampling blocks, hidden width starting at
    `hidden_channels` (paper: 16) and doubled per level.
  * IMPORTANT: matches the paper's *narrow-bottleneck* variant --
    down4 keeps 8*h channels (NOT 16*h), and the up path uses
    nn.Upsample (bilinear) + cat + DoubleConv rather than a learned
    ConvTranspose2d. This is what the reference MARIDA code does.
  * Final 1x1 conv producing the 11 aggregated MARIDA classes.

Training recipe from the paper (see train_marida.py --model unet):
  Adam, lr 2e-4 (x0.1 after epoch 40), batch 5, weighted CE with
  w_c = 1 / ln(1.03 + p_c), random rotations {-90, 0, 90, 180} +
  horizontal flips, ~45 epochs.
"""
import random

import numpy as np
import torch
from torch import nn

# Same reproducibility seeds as the reference implementation.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class Down(nn.Module):
    """Contracting block: MaxPool -> Conv-BN-ReLU -> Conv-BN-ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Expanding block: bilinear upsample (channels unchanged) -> concat
    with the encoder skip -> Conv-BN-ReLU -> Conv-BN-ReLU.

    NOTE: uses nn.Upsample, NOT nn.ConvTranspose2d -- this is what the
    paper's reference U-Net does. Therefore `in_channels` here is
    `skip_channels + decoder_channels` (the sum before the first conv),
    not the previous level's decoder width.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Args:
        input_bands (int): number of input channels (11 raw S2 bands by
            default; + 6 spectral indices and/or + 2 texture features if
            --use_spectral_indices / --use_texture_features are enabled).
        output_classes (int): number of MARIDA classes (11 after water
            aggregation).
        hidden_channels (int): channels of the first block (paper: 16).
        backbone (str): kept for CLI compatibility; only 'none' (the
            plain paper U-Net) is accepted.
    """

    def __init__(self, input_bands=11, output_classes=11, hidden_channels=16, backbone='none'):
        super().__init__()
        if backbone not in (None, 'none'):
            raise ValueError(f"backbone={backbone!r} is not implemented; use 'none' (MARIDA paper U-Net).")

        h = hidden_channels

        # Initial convolution (2 x conv-BN-ReLU at width h).
        self.inc = nn.Sequential(
            nn.Conv2d(input_bands, h, kernel_size=3, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
        )

        # Contracting path. NOTE down4 keeps 8*h channels (narrow bottleneck).
        self.down1 = Down(h,     2 * h)
        self.down2 = Down(2 * h, 4 * h)
        self.down3 = Down(4 * h, 8 * h)
        self.down4 = Down(8 * h, 8 * h)

        # Expanding path. Channel counts before the first conv of each Up
        # are (skip_channels + decoder_channels) because of the concat.
        self.up1 = Up(16 * h, 4 * h)   # skip x4 (8h) + x5 upsampled (8h)
        self.up2 = Up(8  * h, 2 * h)   # skip x3 (4h) + x6 upsampled (4h)
        self.up3 = Up(4  * h,     h)   # skip x2 (2h) + x7 upsampled (2h)
        self.up4 = Up(2  * h,     h)   # skip x1 (h)  + x8 upsampled (h)

        # Final 1x1 head.
        self.outc = nn.Conv2d(h, output_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        return self.outc(x9)