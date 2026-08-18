# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Raw Image Cue Bank & Evidence Fusion Modules for P2 Feature Enhancement."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create 2D Gaussian kernel tensor (1, 1, K, K)."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    grid = coords.repeat(kernel_size, 1)
    g = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.view(1, 1, kernel_size, kernel_size)


class RawImageCueBank(nn.Module):
    """Deterministic, fixed-kernel extraction of color, edge, frequency, and texture cues."""

    def __init__(self):
        super().__init__()
        # Fixed Sobel kernels for edge extraction
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 4.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

        # Fixed Gaussian kernels for frequency separation
        self.register_buffer("gaussian_3", _gaussian_kernel(3, sigma=1.0), persistent=False)
        self.register_buffer("gaussian_7", _gaussian_kernel(7, sigma=2.0), persistent=False)

    def extract_color(self, img0: torch.Tensor) -> torch.Tensor:
        """Extract 4-channel centered chroma/opponent color map at full resolution.

        Cb = -0.3374 R - 0.6626 G + B
        Cr = R - 0.8374 G - 0.1626 B
        O1 = R - G
        O2 = B - 0.5(R + G)
        """
        r, g, b = img0[:, 0:1], img0[:, 1:2], img0[:, 2:3]
        cb = -0.3374 * r - 0.6626 * g + b
        cr = r - 0.8374 * g - 0.1626 * b
        o1 = r - g
        o2 = b - 0.5 * (r + g)
        return torch.cat([cb, cr, o1, o2], dim=1)  # (B, 4, H, W)

    def extract_multi(self, img0: torch.Tensor) -> torch.Tensor:
        """Extract 9-channel multi-cue map at full resolution:

        Color (4ch): Cb, Cr, O1, O2
        Edge (2ch): Gx, Gy via Sobel
        Frequency (2ch): HighPass (Y - G3), DoG (G3 - G7)
        Texture (1ch): Local Variance E[Y^2] - (E[Y])^2
        """
        color4 = self.extract_color(img0)

        # Grayscale Y for geometry/frequency/texture
        r, g, b = img0[:, 0:1], img0[:, 1:2], img0[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b  # (B, 1, H, W)

        # Edge cues
        gx = F.conv2d(y, self.sobel_x, padding=1)
        gy = F.conv2d(y, self.sobel_y, padding=1)

        # Frequency cues
        g3 = F.conv2d(y, self.gaussian_3, padding=1)
        g7 = F.conv2d(y, self.gaussian_7, padding=3)
        high_pass = y - g3
        dog = g3 - g7

        # Texture cue (Local Variance)
        y_sq_avg = F.avg_pool2d(y**2, kernel_size=3, stride=1, padding=1)
        y_avg_sq = (F.avg_pool2d(y, kernel_size=3, stride=1, padding=1))**2
        variance = (y_sq_avg - y_avg_sq).clamp(min=0.0)

        return torch.cat([color4, gx, gy, high_pass, dog, variance], dim=1)  # (B, 9, H, W)

    def forward(self, img0: torch.Tensor, cue_type: str = "color4") -> torch.Tensor:
        if cue_type == "color4":
            cues = self.extract_color(img0)
        elif cue_type == "multi9":
            cues = self.extract_multi(img0)
        else:
            raise ValueError(f"Unknown cue_type: {cue_type}")
        # Deterministic AvgPool4x4 to downsample to stride 4 (128x128)
        return F.avg_pool2d(cues, kernel_size=4, stride=4)


class RawColorSlotFusion(nn.Module):
    """Raw-color channel slot fusion module (A2_color_slots):

    F* = [P_F(F)_{24}, P_B(B)_4, (a_c C_4 + b_c)]
    Total output channels: 32.
    """

    def __init__(self, c_f: int = 24, c_b: int = 4, c_color: int = 4):
        super().__init__()
        self.c_f = c_f
        self.c_b = c_b
        self.c_color = c_color
        self.c2 = c_f + c_b + c_color  # 32

        # 1x1 Conv + BN + SiLU for F (Layer 18, 32ch) and B (Layer 2, 32ch)
        self.proj_f = Conv(32, c_f, 1)
        self.proj_b = Conv(32, c_b, 1)

        # Raw color cue bank
        self.cue_bank = RawImageCueBank()

        # Learnable affine transformation per color channel, initialized to a=1, b=0
        self.scale = nn.Parameter(torch.ones(1, c_color, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, c_color, 1, 1))

    def forward(self, x: list[torch.Tensor], img0: torch.Tensor) -> torch.Tensor:
        f_feat, b_feat = x[0], x[1]  # F = Layer 18, B = Layer 2
        f24 = self.proj_f(f_feat)
        b4 = self.proj_b(b_feat)

        # Extract stride-4 color map
        c4_raw = self.cue_bank(img0, cue_type="color4")
        c4 = self.scale * c4_raw + self.shift

        return torch.cat([f24, b4, c4], dim=1)  # (B, 32, 128, 128)


class MultiCueEvidenceFusion(nn.Module):
    """Multi-cue evidence formation module (B_color_formation & B_multi_formation):

    F* = [P_F(F)_{24}, E_8]
    where E_8 = B_8 + ΔE_8, and ΔE_8 = φ([B_8, C_multi]) zero-initialized.
    Total output channels: 32.
    """

    def __init__(self, c_f: int = 24, c_b: int = 8, cue_type: str = "multi9"):
        super().__init__()
        self.c_f = c_f
        self.c_b = c_b
        self.cue_type = cue_type
        self.c2 = c_f + c_b  # 32

        n_cues = 4 if cue_type == "color4" else 9

        self.proj_f = Conv(32, c_f, 1)
        self.proj_b = Conv(32, c_b, 1)

        self.cue_bank = RawImageCueBank()

        # Evidential formation block φ: 1x1 Conv -> DWConv3x3 -> 1x1 Conv (zero-initialized)
        in_dim = c_b + n_cues  # 8 + 4 = 12, or 8 + 9 = 17
        self.conv1 = Conv(in_dim, 16, 1)
        self.dwconv = Conv(16, 16, 3, p=1, g=16)

        # Pure linear 1x1 conv to 8 channels, zero-initialized
        self.conv2 = nn.Conv2d(16, c_b, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: list[torch.Tensor], img0: torch.Tensor) -> torch.Tensor:
        f_feat, b_feat = x[0], x[1]
        f24 = self.proj_f(f_feat)
        b8 = self.proj_b(b_feat)

        cues = self.cue_bank(img0, cue_type=self.cue_type)
        u = torch.cat([b8, cues], dim=1)  # (B, in_dim, 128, 128)

        delta_e = self.conv2(self.dwconv(self.conv1(u)))
        e8 = b8 + delta_e

        return torch.cat([f24, e8], dim=1)  # (B, 32, 128, 128)
