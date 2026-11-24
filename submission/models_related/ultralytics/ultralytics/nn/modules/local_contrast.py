# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Local-contrast feature-formation modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2f
from .conv import Conv


class LocalContrastBasisStem(nn.Module):
    """Multi-scale local-contrast basis stem for tiny-object feature formation.

    The main path reproduces the standard YOLOv8n stem through P2. In parallel,
    an auxiliary path forms two full-resolution local-reference bases at 9x9 and
    17x17 support. Each basis contains a signed RGB residual plus its RGB-vector
    magnitude. A shared encoder is intentionally used at both scales so that
    channel-wise cross-scale product and discrepancy terms are semantically valid.

    ``mode='raw'`` keeps exactly the same trainable graph but feeds duplicated raw
    RGB + RGB magnitude to the auxiliary path. It is the matched-capacity control.

    There is no sigmoid/tanh/softmax gating and no residual feature correction.
    The fused P2 tensor becomes the actual backbone state that is downsampled into P3.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        mode: str = "relative",
        k_small: int = 9,
        k_large: int = 17,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()
        if mode not in {"relative", "relative_no_cross", "raw", "relative_r_only", "raw_independent"}:
            raise ValueError(f"invalid LocalContrastBasisStem mode: {mode!r}")
        if k_small % 2 == 0 or k_large % 2 == 0 or k_small >= k_large:
            raise ValueError("LocalContrastBasisStem requires odd k_small < k_large")

        self.mode = mode
        self.k_small = int(k_small)
        self.k_large = int(k_large)
        self.fusion_mode = fusion_mode.lower()

        # Main path: same P2-producing stem as the YOLOv8n baseline.
        c_mid = max(c2 // 2, 8)
        self.main_cv1 = Conv(c1, c_mid, 3, 2)
        self.main_cv2 = Conv(c_mid, c2, 3, 2)
        self.main_c2f = C2f(c2, c2, n=1, shortcut=True)

        # Relative path. Each local basis has 3 signed RGB residual channels +
        # one RGB-vector magnitude channel. The same encoder is reused for 9x9/17x17.
        c_rel = max(c2 // 4, 8)
        if self.mode == "raw_independent":
            self.rel_encoder_small = nn.Sequential(
                Conv(c1 + 1, c_rel, 3, 2),
                Conv(c_rel, c_rel, 3, 2),
                C2f(c_rel, c_rel, n=1, shortcut=True),
            )
            self.rel_encoder_large = nn.Sequential(
                Conv(c1 + 1, c_rel, 3, 2),
                Conv(c_rel, c_rel, 3, 2),
                C2f(c_rel, c_rel, n=1, shortcut=True),
            )
        else:
            self.rel_encoder = nn.Sequential(
                Conv(c1 + 1, c_rel, 3, 2),
                Conv(c_rel, c_rel, 3, 2),
                C2f(c_rel, c_rel, n=1, shortcut=True),
            )

        # Cross-scale basis contains short-scale state, long-scale state,
        # channel-wise agreement, and channel-wise discrepancy.
        in_scale = 2 * c_rel if self.mode == "relative_r_only" else 4 * c_rel
        self.scale_formation = C2f(in_scale, c2, n=1, shortcut=False)

        # Joint formation is intentionally not an additive adapter. The output is
        # a newly formed P2 representation and is the sole state propagated onward.
        self.joint_formation = C2f(2 * c2, c2, n=1, shortcut=False)
        if self.fusion_mode == "ffm":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc_p = nn.Conv2d(c2, c2, 1)
            self.fc_q = nn.Conv2d(c2, c2, 1)
            self.conv_out = Conv(c2 * 3, c2, 1)

    @staticmethod
    def _rgb_magnitude(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-8)

    def _local_basis(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        # Dense box pooling is faster than two separable pooling launches on the target CUDA stack.
        local_mean = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        residual = x - local_mean
        magnitude = self._rgb_magnitude(residual)
        return torch.cat((residual, magnitude), dim=1)

    def _raw_basis(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self._rgb_magnitude(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_c2f(self.main_cv2(self.main_cv1(x)))

        if self.mode in {"relative", "relative_no_cross"}:
            basis_small = self._local_basis(x, self.k_small)
            basis_large = self._local_basis(x, self.k_large)
        else:
            basis_small = self._raw_basis(x)
            basis_large = self._raw_basis(x)

        if self.mode == "raw_independent":
            rel_small = self.rel_encoder_small(basis_small)
            rel_large = self.rel_encoder_large(basis_large)
        else:
            rel_small = self.rel_encoder(basis_small)
            rel_large = self.rel_encoder(basis_large)
        if self.mode == "relative_no_cross":
            # Same 4*C width and active parameters, but no explicit cross-scale products/differences.
            scale_state = torch.cat((rel_small, rel_large, rel_small, rel_large), dim=1)
        elif self.mode == "relative_r_only":
            # Direct concat of rel_small and rel_large without product/diff basis
            scale_state = torch.cat((rel_small, rel_large), dim=1)
        else:
            scale_state = torch.cat(
                (
                    rel_small,
                    rel_large,
                    rel_small * rel_large,
                    torch.abs(rel_small - rel_large),
                ),
                dim=1,
            )
        relative = self.scale_formation(scale_state)
        self.last_D = (1.0 - F.cosine_similarity(main, relative, dim=1)).detach()
        if getattr(self, "fusion_mode", "concat") == "ffm":
            p = self.fc_p(self.gap(relative))
            q = self.fc_q(self.gap(main))
            S_B = (q * main).sum(dim=1, keepdim=True)
            B_hat = p * S_B
            return self.conv_out(torch.cat([main, B_hat, relative], dim=1))
        elif getattr(self, "fusion_mode", "concat") == "r_only":
            return relative
        else:
            return self.joint_formation(torch.cat((main, relative), dim=1))


class SingleContrastFormationStem(nn.Module):
    """Single-Scale Local Contrast Formation Stem (LCF-Stem)."""

    def __init__(self, c1: int, c2: int, k: int = 17, mode: str = "relative", use_form: bool = True) -> None:
        super().__init__()
        self.k = int(k)
        self.mode = mode
        self.use_form = use_form

        c_hidden = max(c2 // 4, 8)
        self.enc = nn.Sequential(
            Conv(c1, c_hidden, 3, 2),
            Conv(c_hidden, c_hidden, 3, 2),
            C2f(c_hidden, c_hidden, n=1, shortcut=True),
        )

        if self.use_form:
            self.form = C2f(c_hidden, c2, n=1, shortcut=False)
        else:
            self.form = nn.Conv2d(c_hidden, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "relative":
            pad = self.k // 2
            mean = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=self.k, stride=1)
            contrast = x - mean
        else:
            contrast = x

        z = self.enc(contrast)
        p2 = self.form(z)
        return p2


class SidecarResidualFusionStem(nn.Module):
    """Sidecar P2 Fusion Stem supporting 4 variants (linear_residual, joint_residual, norm_joint_residual, replace_joint)"""

    def __init__(self, c1: int, c2: int, k: int = 17, mode: str = "joint_residual") -> None:
        super().__init__()
        self.mode = mode
        self.k = int(k)

        # Main Path: YOLO standard P2 stem
        c_mid = max(c2 // 2, 8)
        self.main_cv1 = Conv(c1, c_mid, 3, 2)
        self.main_cv2 = Conv(c_mid, c2, 3, 2)
        self.main_c2f = C2f(c2, c2, n=1, shortcut=True)

        # Sidecar Path: Stride-4 contrast encoder
        c_rel = max(c2 // 4, 8)
        self.rel_encoder = nn.Sequential(
            Conv(c1, c_rel, 3, 2),
            Conv(c_rel, c_rel, 3, 2),
            C2f(c_rel, c_rel, n=1, shortcut=True),
        )

        # Fusion operators
        if self.mode == "linear_residual":
            self.proj_r = nn.Conv2d(c_rel, c2, 1)
            nn.init.zeros_(self.proj_r.weight)
            nn.init.zeros_(self.proj_r.bias)
        elif self.mode in {"joint_residual", "replace_joint"}:
            self.fusion_c2f = C2f(c2 + c_rel, c2, n=1, shortcut=False)
            if self.mode == "joint_residual":
                self.proj_joint = nn.Conv2d(c2, c2, 1)
                nn.init.zeros_(self.proj_joint.weight)
                nn.init.zeros_(self.proj_joint.bias)
        elif self.mode == "norm_joint_residual":
            self.gn_m = nn.GroupNorm(4 if c2 % 4 == 0 else 2, c2)
            self.gn_r = nn.GroupNorm(4 if c_rel % 4 == 0 else 2, c_rel)
            self.fusion_c2f = C2f(c2 + c_rel, c2, n=1, shortcut=False)
            self.proj_joint = nn.Conv2d(c2, c2, 1)
            nn.init.zeros_(self.proj_joint.weight)
            nn.init.zeros_(self.proj_joint.bias)
        else:
            raise ValueError(f"invalid SidecarResidualFusionStem mode: {self.mode!r}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Main Path
        m = self.main_c2f(self.main_cv2(self.main_cv1(x)))

        # 2. Sidecar Contrast Path
        pad = self.k // 2
        mean = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=self.k, stride=1)
        contrast = x - mean
        r = self.rel_encoder(contrast)

        # 3. Fusion Output F
        if self.mode == "linear_residual":
            f = m + self.proj_r(r)
        elif self.mode == "joint_residual":
            joint = torch.cat((m, r), dim=1)
            f = m + self.proj_joint(self.fusion_c2f(joint))
        elif self.mode == "norm_joint_residual":
            m_norm = self.gn_m(m)
            r_norm = self.gn_r(r)
            joint = torch.cat((m_norm, r_norm), dim=1)
            f = m + self.proj_joint(self.fusion_c2f(joint))
        elif self.mode == "replace_joint":
            joint = torch.cat((m, r), dim=1)
            f = self.fusion_c2f(joint)

        # Return (M, F) tuple. Downstream Backbone layers take M; Downstream FPN takes F.
        return m, f
