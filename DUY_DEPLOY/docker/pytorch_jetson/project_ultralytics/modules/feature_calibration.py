"""Project-specific feature calibration and perturbation modules.

These modules were extracted from the legacy fork and intentionally depend
only on PyTorch, so they can be used with clean upstream Ultralytics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class P2AmplitudeCalibrator(nn.Module):
    """
    Computes a single scalar scaling factor alpha per image based on global statistics of P2.
    Preserves all spatial and channel correlation relationships.
    Initialized as an identity mapping (alpha=1.0) at step 0.
    """
    def __init__(self, channels=None, hidden_dim=16, min_val=0.25, max_val=1.75) -> None:
        super().__init__()
        self.min_val = min_val
        self.scale = max_val - min_val

        # 4 features: Mean Absolute, RMS, Std, Channel Dispersion Std
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize last layer to zeros to ensure initial output is 0.0 (identity mapping)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. Mean absolute activation
        mean_abs = torch.mean(torch.abs(x), dim=[1, 2, 3])  # [B]

        # 2. RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=[1, 2, 3]) + 1e-6)  # [B]

        # 3. Spatial/channel Standard Deviation
        std = torch.std(x, dim=[1, 2, 3])  # [B]

        # 4. Channel dispersion (std of channel means)
        channel_means = torch.mean(x, dim=[2, 3])  # [B, C]
        chan_disp = torch.std(channel_means, dim=1)  # [B]

        # Stack stats
        stats = torch.stack([mean_abs, rms, std, chan_disp], dim=1)  # [B, 4]

        # Predict z, apply sigmoid and scale to range [min_val, max_val]
        z = self.mlp(stats)  # [B, 1]
        alpha = self.min_val + self.scale * torch.sigmoid(z)  # [B, 1]

        # Apply scaling: [B, 1, 1, 1] * [B, C, H, W]
        return x * alpha.view(B, 1, 1, 1)


class LearnableGlobalScalar(nn.Module):
    """
    A single learnable scalar parameter shared across all images and channels.
    Initialized as an identity mapping (alpha=1.0) at step 0.
    """
    def __init__(self, channels=None) -> None:
        super().__init__()
        # Initialize parameter to 0.0, passing through sigmoid:
        # alpha = 0.25 + 1.5 * sigmoid(param) -> param=0.0 means alpha=1.0
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 0.25 + 1.5 * torch.sigmoid(self.param)
        return x * alpha


class AmplitudePerturbation(nn.Module):
    """
    Applies random scale perturbation to channel amplitudes during training.
    Bypassed during evaluation (clean inference).
    """
    def __init__(self, mode: str = "channel", scale_range=(0.7, 1.3), chan_range=(0.95, 1.05)) -> None:
        super().__init__()
        self.mode = mode  # "image", "channel", or "none"
        self.scale_range = scale_range
        self.chan_range = chan_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mode == "none":
            return x

        B, C, H, W = x.shape
        if self.mode == "image":
            # Single random scalar per image
            s = x.new_empty(B, 1, 1, 1).uniform_(*self.scale_range)
            return x * s
        elif self.mode == "channel":
            # Random scaling factor per channel per image
            s = x.new_empty(B, C, 1, 1).uniform_(*self.chan_range)
            return x * s
        return x


class MatchedChannelPerturbation(nn.Module):
    """0-param channel perturbation matched to train-split GAP gate statistics."""

    def __init__(self, channels=None, mu=0.4514, sigma_delta=0.05, q01=0.0, q99=1.0, eps=1e-6) -> None:
        super().__init__()
        self.mu = float(mu)
        self.sigma_delta = float(sigma_delta)
        self.q01 = float(q01)
        self.q99 = float(q99)
        self.eps = float(eps)
        self.last_gate_stats = {}
        self._logged_modes = set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            b, c, _, _ = x.shape
            z = torch.randn(b, c, 1, 1, device=x.device, dtype=x.dtype)
            z = z - z.mean(dim=1, keepdim=True)
            z = z / (z.std(dim=1, keepdim=True, correction=0) + self.eps)
            gate = self.mu + self.sigma_delta * z
            gate = gate.clamp(self.q01, self.q99)
        else:
            gate = x.new_tensor(self.mu)
        self.last_gate_stats = self._gate_stats(gate)
        mode = "train" if self.training else "eval"
        if mode not in self._logged_modes:
            s = self.last_gate_stats
            print(
                f"MatchedChannelPerturbation {mode}: "
                f"gate_mean={s['gate_mean']:.6f}, "
                f"gate_channel_std={s['gate_channel_std']:.6f}, "
                f"gate_min={s['gate_min']:.6f}, gate_max={s['gate_max']:.6f}"
            )
            self._logged_modes.add(mode)
        return x * gate

    @staticmethod
    def _gate_stats(gate: torch.Tensor) -> dict:
        g = gate.detach().float()
        return {
            "gate_mean": float(g.mean()),
            "gate_channel_std": float(g.std(dim=1, correction=0).mean()) if g.ndim == 4 and g.shape[1] > 1 else 0.0,
            "gate_min": float(g.min()),
            "gate_max": float(g.max()),
        }


class ResidualDWConv(nn.Module):
    """Residual depthwise local mixer with optional partial-channel path."""

    def __init__(self, channels=None, k=5, alpha=0.1, partial_ratio=1.0) -> None:
        super().__init__()
        if not 0 < partial_ratio <= 1:
            raise ValueError(f"partial_ratio must be in (0, 1], got {partial_ratio}")
        active_channels = max(1, int(round(channels * partial_ratio)))
        self.channels = channels
        self.active_channels = active_channels
        self.k = int(k)
        self.partial_ratio = float(partial_ratio)
        self.dw = nn.Conv2d(active_channels, active_channels, self.k, padding=self.k // 2, groups=active_channels, bias=False)
        self.bn = nn.BatchNorm2d(active_channels)
        self.act = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_active, x_skip = x.split((self.active_channels, self.channels - self.active_channels), dim=1)
        y_active = x_active + self.alpha.to(dtype=x.dtype) * self.act(self.bn(self.dw(x_active)))
        return y_active if x_skip.shape[1] == 0 else torch.cat((y_active, x_skip), dim=1)


class ResidualDWConv5(ResidualDWConv):
    """Backward-compatible depthwise 5x5 residual mixer."""

    def __init__(self, channels=None, alpha=0.1) -> None:
        super().__init__(channels, k=5, alpha=alpha, partial_ratio=1.0)


class P2FeatureProbe(nn.Module):
    """Tiny identity-initialized P2 feature probes before Detect."""

    def __init__(self, channels=None, mode="context", eps=1e-6) -> None:
        super().__init__()
        if mode not in {"context", "residual", "energy", "mean_center", "std_norm", "location_rms", "global_add"}:
            raise ValueError(f"Unsupported P2FeatureProbe mode: {mode!r}")
        self.mode = mode
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1))
        self.context = nn.Conv2d(channels, channels, 1, bias=False) if mode == "context" else None
        self.global_add = nn.Conv2d(channels, channels, 1, bias=False) if mode == "global_add" else None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)
        if self.global_add is not None:
            nn.init.zeros_(self.global_add.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "context":
            delta = self.context(torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1))
        elif self.mode == "residual":
            delta = x - torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
        elif self.mode == "energy":
            rms = torch.sqrt(x.square().mean(dim=1, keepdim=True) + self.eps)
            delta = x / rms
        elif self.mode == "mean_center":
            delta = -x.mean(dim=(2, 3), keepdim=True)
        elif self.mode == "std_norm":
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True, correction=0)
            delta = (x - mean) / (std + self.eps)
        elif self.mode == "location_rms":
            rms = torch.sqrt(x.square().mean(dim=1, keepdim=True) + self.eps)
            delta = x / rms
        else:
            delta = self.global_add(torch.nn.functional.adaptive_avg_pool2d(x, 1))
        return x + self.gamma.to(dtype=x.dtype) * delta
