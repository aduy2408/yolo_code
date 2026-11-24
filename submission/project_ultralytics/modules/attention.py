"""Project-specific KV-compressed attention modules.

Extracted from the legacy Ultralytics fork. The implementations use only
PyTorch plus clean upstream ``Conv`` and ``LayerNorm2d`` primitives.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.transformer import LayerNorm2d

def _choose_attention_heads(channels: int, requested_heads: int) -> int:
    """Pick a valid attention head count that divides channels."""
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _group_weight_compress(x: torch.Tensor, sr_ratio: int, scorer: nn.Linear) -> torch.Tensor:
    """Compress non-overlapping spatial groups with learned softmax token weights."""
    if sr_ratio <= 1:
        return x
    b, c, h, w = x.shape
    pad_h = (sr_ratio - h % sr_ratio) % sr_ratio
    pad_w = (sr_ratio - w % sr_ratio) % sr_ratio
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = x.shape[-2:]
    gh, gw = hp // sr_ratio, wp // sr_ratio
    tokens = x.view(b, c, gh, sr_ratio, gw, sr_ratio).permute(0, 2, 4, 3, 5, 1).contiguous()
    tokens = tokens.view(b, gh, gw, sr_ratio * sr_ratio, c)
    weights = scorer(tokens).softmax(dim=3)
    compressed = (tokens * weights).sum(dim=3)
    return compressed.permute(0, 3, 1, 2).contiguous()


class KVCompressedAttention(nn.Module):
    """Self-attention with full-resolution queries and spatially compressed keys/values.

    Supports multiple K/V spatial compression modes:
    - avg: AvgPool + GroupNorm for K and V.
    - avg_dwk/dwconv: AvgPool + DWConv + GroupNorm for K, AvgPool + GroupNorm for V.
    - dw_stride: stride depthwise Conv + GroupNorm for K and V.
    - group_weight: learned softmax weighting inside each sr_ratio x sr_ratio group.
    Attention is computed via ``F.scaled_dot_product_attention`` which automatically
    dispatches to FlashAttention v2 on supported CUDA hardware.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        residual: bool = True,
    ):
        """Initialize KV-compressed attention.

        Args:
            c1: Input channels.
            c2: Output channels.
            num_heads: Requested attention heads. Reduced if it does not divide c2.
            sr_ratio: Spatial compression ratio for K/V tokens.
            mode: ``avg``, ``avg_dwk``, ``dw_stride``, ``dwconv``, or ``group_weight`` compression.
            attn_drop: Dropout probability applied to attention weights during training.
            residual: Whether to add the projected attention output back to the input projection.
        """
        super().__init__()
        if mode == "dwconv":
            mode = "avg_dwk"
        if mode not in {"avg", "avg_dwk", "dw_stride", "group_weight"}:
            raise ValueError(f"Unsupported KV compression mode: {mode}")

        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = max(1, int(sr_ratio))
        self.mode = mode
        self.attn_drop_p = attn_drop
        self.residual = residual

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)  # stabilize Q logits before SDPA

        if self.sr_ratio > 1 and self.mode == "avg":
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "avg_dwk":
            # No activation - K must stay linear for well-formed attention logits.
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "dw_stride":
            self.k_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
        else:
            self.k_compress = nn.Identity()
            self.v_compress = nn.Identity()

        # Separate linear projections for K and V (no shared kv conv, no activation)
        self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)

        # group_weight path keeps its own scorer (unchanged)
        self.group_score = nn.Linear(c2, 1) if self.mode == "group_weight" and self.sr_ratio > 1 else None
        # Shared kv conv kept for group_weight compatibility
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False) if self.mode == "group_weight" else None

        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)
        if self.residual:
            nn.init.zeros_(self.proj_bn.weight)
            nn.init.zeros_(self.proj_bn.bias)

    def _compress_group_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Compress each sr_ratio x sr_ratio token group with learned softmax weights."""
        return _group_weight_compress(x, self.sr_ratio, self.group_score)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KV-compressed attention and return a BCHW tensor."""
        x = self.input_proj(x)
        b, c, h, w = x.shape

        # Q: full-resolution, normalized per head
        q = self.q(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, H*W, hd]
        q = self.q_norm(q)

        if self.mode == "group_weight" and self.group_score is not None:
            # group_weight path: unchanged (shared kv compress)
            kv_source = self._compress_group_weight(x)
            kv = self.kv(kv_source).flatten(2).transpose(1, 2)
            kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            # Non-group-weight paths use separate K and V compression and projection.
            k_src = self.k_compress(x)  # [B, C, H/sr, W/sr]
            v_src = self.v_compress(x)  # [B, C, H/sr, W/sr]
            k = self.k_proj(k_src)  # no activation
            v = self.v_proj(v_src)  # no activation
            # reshape to [B, nh, tokens, hd]
            def _to_heads(t):
                n = t.shape[2] * t.shape[3]
                return t.flatten(2).transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = _to_heads(k)
            v = _to_heads(v)

        # Flash Attention via PyTorch SDPA (dispatches to FlashAttn v2 on CUDA)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj_bn(self.proj(out))
        return x + out if self.residual else out


class ReceptanceKVCompressedAttention(KVCompressedAttention):
    """KV-compressed attention with a per-channel, per-location receptance gate.

    The inherited attention core is intentionally left unchanged.  The gate is
    applied to the raw SDPA result before the output projection, and the
    zero-initialized projection BatchNorm preserves exact residual identity at
    initialization.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "group_weight",
        attn_drop: float = 0.0,
        residual: bool = True,
    ):
        super().__init__(c1, c2, num_heads, sr_ratio, mode, attn_drop, residual)
        self.receptance = nn.Conv2d(c2, c2, 1, bias=True)
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.receptance.bias)
        self.capture_receptance = False
        self.last_receptance = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unchanged KVCA attention core followed by receptance."""
        x = self.input_proj(x)
        b, c, h, w = x.shape

        q = self.q(x).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        if self.mode == "group_weight" and self.group_score is not None:
            kv_source = self._compress_group_weight(x)
            kv = self.kv(kv_source).flatten(2).transpose(1, 2)
            kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            k_src = self.k_compress(x)
            v_src = self.v_compress(x)
            k = self.k_proj(k_src)
            v = self.v_proj(v_src)

            def _to_heads(t):
                n = t.shape[2] * t.shape[3]
                return t.flatten(2).transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k = _to_heads(k)
            v = _to_heads(v)

        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )
        a = a.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        g = torch.sigmoid(self.receptance(x))
        if self.capture_receptance:
            self.last_receptance = g.detach()
        a = g * a
        out = self.proj_bn(self.proj(a))
        return x + out if self.residual else out

    def receptance_statistics(self) -> dict[str, float]:
        """Summarize the most recently captured gate without retaining its graph."""
        if self.last_receptance is None:
            raise RuntimeError("No receptance captured; set capture_receptance=True and run a forward pass first")
        g = self.last_receptance.float()
        return {
            "mean": float(g.mean().cpu()),
            "std": float(g.std(unbiased=False).cpu()),
            "p10": float(torch.quantile(g, 0.10).cpu()),
            "p50": float(torch.quantile(g, 0.50).cpu()),
            "p90": float(torch.quantile(g, 0.90).cpu()),
            "fraction_below_0.25": float((g < 0.25).float().mean().cpu()),
            "fraction_above_0.75": float((g > 0.75).float().mean().cpu()),
        }


class GlobalChannelContextCalibration(nn.Module):
    """Calibrate local features using global cross-channel relationships without spatial content transport."""

    def __init__(
        self,
        c1: int,
        c2: int,
        sr_ratio: int = 8,
        temperature: float = 0.07,
        alpha_init: float = 0.0,
    ):
        """Initialize learned group compression and global channel-affinity gating."""
        super().__init__()
        if isinstance(sr_ratio, bool) or int(sr_ratio) != sr_ratio or sr_ratio <= 0:
            raise ValueError(f"sr_ratio must be a positive integer, got {sr_ratio}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.c2 = c2
        self.sr_ratio = int(sr_ratio)
        self.temperature = float(temperature)
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.group_score = nn.Linear(c2, 1)
        self.q = nn.Conv1d(c2, c2, 1, bias=False)
        self.k = nn.Conv1d(c2, c2, 1, bias=False)
        self.v = nn.Conv1d(c2, c2, 1, bias=False)
        self.gate_proj = nn.Linear(c2, c2)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_gate_shape = None

    def _compress_group_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the matched KVCA compression for validation and diagnostics."""
        return _group_weight_compress(x, self.sr_ratio, self.group_score)

    def channel_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Return a global gate of shape B,C,1,1."""
        tokens = self._compress_group_weight(x).flatten(2)  # B,C,N
        q = F.normalize(self.q(tokens), dim=-1)
        k = F.normalize(self.k(tokens), dim=-1)
        affinity = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.temperature, dim=-1)
        descriptor = self.v(tokens).mean(dim=-1)
        context = torch.matmul(affinity, descriptor.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(self.gate_proj(context)).unsqueeze(-1).unsqueeze(-1)

    def analytical_macs(self, height: int, width: int) -> int:
        """Return custom-operator MACs for one sample, including compression and feature modulation."""
        groups = math.ceil(height / self.sr_ratio) * math.ceil(width / self.sr_ratio)
        c = self.c2
        return height * width * c + 3 * groups * c * c + 2 * groups * c * c + 2 * c * c + height * width * c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative channel calibration while preserving each location's local feature."""
        x = self.input_proj(x)
        gate = self.channel_gate(x)
        self.last_gate_shape = tuple(gate.shape)
        return x * (1.0 + self.alpha * gate)


class PatchKVCompressedAttention(KVCompressedAttention):
    """Group-weight KV attention restricted to a local compressed-grid neighborhood."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 8,
        patch_radius: int = 1,
        residual: bool = True,
    ):
        """Initialize Patch-KVCA with the same projections as group-weight KVCA."""
        if int(patch_radius) != patch_radius or patch_radius < 0:
            raise ValueError(f"patch_radius must be a non-negative integer, got {patch_radius}")
        super().__init__(c1, c2, num_heads, sr_ratio, mode="group_weight", residual=residual)
        self.patch_radius = int(patch_radius)

    @staticmethod
    def _validity_mask(gh: int, gw: int, radius: int, device: torch.device) -> torch.Tensor:
        """Return valid local keys for every compressed-grid location."""
        kernel_size = 2 * radius + 1
        valid = F.unfold(
            torch.ones(1, 1, gh, gw, dtype=torch.bool, device=device),
            kernel_size=kernel_size,
            padding=radius,
        )
        return valid.transpose(1, 2)  # [1, groups, local keys]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local Patch-KVCA and return a BCHW tensor."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        sr = self.sr_ratio
        pad_h = (sr - h % sr) % sr
        pad_w = (sr - w % sr) % sr
        hp, wp = h + pad_h, w + pad_w
        gh, gw = hp // sr, wp // sr

        # Keep full-resolution Q; padding only fills incomplete edge groups and is cropped after attention.
        q = self.q(x)
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h))
        q = q.reshape(b, self.num_heads, self.head_dim, gh, sr, gw, sr)
        q = q.permute(0, 1, 3, 5, 4, 6, 2).reshape(
            b, self.num_heads, gh * gw, sr * sr, self.head_dim
        )
        q = self.q_norm(q)

        kv_source = self._compress_group_weight(x)
        kv = self.kv(kv_source).reshape(b, 2, self.num_heads, self.head_dim, gh, gw)
        k_map, v_map = kv[:, 0], kv[:, 1]
        kernel_size = 2 * self.patch_radius + 1

        def _local_patches(t: torch.Tensor) -> torch.Tensor:
            patches = F.unfold(
                t.reshape(b, c, gh, gw),
                kernel_size=kernel_size,
                padding=self.patch_radius,
            )
            return patches.reshape(b, self.num_heads, self.head_dim, kernel_size**2, gh * gw).permute(
                0, 1, 4, 3, 2
            )

        k = _local_patches(k_map)
        v = _local_patches(v_map)
        local_batch = b * self.num_heads * gh * gw
        q = q.reshape(local_batch, sr * sr, self.head_dim)
        k = k.reshape(local_batch, kernel_size**2, self.head_dim)
        v = v.reshape(local_batch, kernel_size**2, self.head_dim)
        mask = self._validity_mask(gh, gw, self.patch_radius, x.device)
        mask = mask[:, None].expand(b, self.num_heads, gh * gw, kernel_size**2)
        mask = mask.reshape(local_batch, 1, kernel_size**2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = out.reshape(b, self.num_heads, gh, gw, sr, sr, self.head_dim)
        out = out.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, c, hp, wp)[..., :h, :w]
        out = self.proj_bn(self.proj(out))
        return x + out if self.residual else out


class FullSelfAttention(KVCompressedAttention):
    """Full-resolution self-attention using the same projections as group-weight KVCA."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, residual: bool = True):
        """Initialize full P2 self-attention without spatial K/V compression."""
        super().__init__(c1, c2, num_heads, sr_ratio=8, mode="group_weight", residual=residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-resolution self-attention and return a BCHW tensor."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        kv = self.kv(x).flatten(2).transpose(1, 2)
        kv = kv.reshape(b, h * w, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1], scale=self.scale)
        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj_bn(self.proj(out))
        return x + out if self.residual else out


class KVCompressedTransformerEncoder(nn.Module):
    """Pre-norm transformer encoder block using KV-compressed self-attention and a DW-PW convolutional FFN.

    FFN structure: PW-expand → DW-spatial-mix → PW-project, giving each token access to its
    8-connected spatial neighborhood while keeping the channel mixing role of the outer PWs.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        """Initialize LayerNorm-KVCA and LayerNorm-DW-FFN residual branches.

        Args:
            c1: Input channels.
            c2: Output channels.
            num_heads: Requested attention heads.
            sr_ratio: Spatial compression ratio passed to KVCompressedAttention.
            mode: KV compression mode (``dwconv`` or ``group_weight``).
            attn_drop: Attention dropout probability (training only).
            mlp_ratio: Hidden dim multiplier for the FFN.
        """
        super().__init__()
        hidden = max(c2, int(c2 * mlp_ratio))
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.norm1 = LayerNorm2d(c2)
        self.attn = KVCompressedAttention(c2, c2, num_heads, sr_ratio, mode, attn_drop=attn_drop, residual=False)
        self.norm2 = LayerNorm2d(c2)
        self.ffn = nn.Sequential(
            Conv(c2, hidden, 1, 1),                    # PW: channel expand
            Conv(hidden, hidden, 3, 1, g=hidden),      # DW: spatial mix in 3×3 neighborhood
            Conv(hidden, c2, 1, 1, act=False),         # PW: channel project
        )
        # Zero-initialize attention projection and FFN projection BNs for identity behavior
        nn.init.zeros_(self.attn.proj_bn.weight)
        nn.init.zeros_(self.attn.proj_bn.bias)
        nn.init.zeros_(self.ffn[-1].bn.weight)
        nn.init.zeros_(self.ffn[-1].bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LN -> KVCA -> residual, then LN -> DW-FFN -> residual."""
        x = self.input_proj(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class KVCompressedAttentionPartial(nn.Module):
    """PSA-style partial KV-compressed attention.

    Splits input channels in half: one half passes through ``KVCompressedAttention``
    (with a residual connection), the other half bypasses unchanged.  The two halves
    are concatenated and projected back to ``c2`` channels with a pointwise Conv.

    Benefits at high-resolution feature maps (P2/P3):
    - Attention head_dim is halved → compute reduced ~50 %.
    - Bypass half retains fine-grained local texture untouched by attention.
    - Parameter overhead is small: only an extra ``Conv(c2, c2, 1)`` output projection.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
    ):
        """Initialize partial KVCA.

        Args:
            c1: Input channels.
            c2: Output channels (must be even).
            num_heads: Attention heads for the *attention* half; clipped to c2//2.
            sr_ratio: Spatial compression ratio passed to ``KVCompressedAttention``.
            mode: KV compression mode (``dwconv`` or ``group_weight``).
            attn_drop: Attention dropout probability (training only).
        """
        super().__init__()
        if c2 % 2 != 0:
            raise ValueError(f"KVCompressedAttentionPartial requires even c2, got {c2}")
        c_attn = c2 // 2
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.attn = KVCompressedAttention(
            c_attn,
            c_attn,
            num_heads=max(1, num_heads // 2),  # half heads for half channels
            sr_ratio=sr_ratio,
            mode=mode,
            attn_drop=attn_drop,
            residual=True,
        )
        # Pointwise mix after concat – no activation to stay linear
        self.out_proj = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KVCA on first half of channels, bypass second half, mix outputs."""
        x = self.input_proj(x)
        x_attn, x_bypass = x.chunk(2, dim=1)
        x_attn = self.attn(x_attn)
        return self.out_proj(torch.cat([x_attn, x_bypass], dim=1))


class SurgicalPartialKVCompressedAttention(nn.Module):
    """Identity-safe partial KVCA for a surgical ablation.

    Half of the channels use KVCA and half bypass it.  Unlike the legacy
    ``KVCompressedAttentionPartial``, this probe has no learned post-concat
    projection, so its default KVCA residual initialization is an exact
    identity when ``c1 == c2``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if c2 % 2 != 0:
            raise ValueError(f"SurgicalPartialKVCompressedAttention requires even c2, got {c2}")
        c_attn = c2 // 2
        self.c2 = c2
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.attn = KVCompressedAttention(
            c_attn,
            c_attn,
            num_heads=max(1, num_heads // 2),
            sr_ratio=sr_ratio,
            mode=mode,
            attn_drop=attn_drop,
            residual=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KVCA to the first channel half and bypass the second half."""
        x = self.input_proj(x)
        x_attn, x_bypass = x.chunk(2, dim=1)
        return torch.cat([self.attn(x_attn), x_bypass], dim=1)


class _TopKGroupKVAttentionBase(nn.Module):
    """Shared utilities for top-k grouped K/V attention blocks."""

    def __init__(self, c1: int, c2: int, num_heads: int, group_size: int):
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.group_size = max(1, int(group_size))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False)
        self.score = nn.Conv2d(c2, 1, 3, padding=1, bias=True)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to a multiple and return the original H/W."""
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def _to_groups(self, x: torch.Tensor, group_h: int, group_w: int) -> torch.Tensor:
        """Convert BCHW into B x groups x group_tokens x C."""
        b, c, _, _ = x.shape
        g = self.group_size
        return (
            x.view(b, c, group_h, g, group_w, g)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, group_h * group_w, g * g, c)
        )

    @staticmethod
    def _to_regions(x: torch.Tensor, region_size: int, region_h: int, region_w: int) -> torch.Tensor:
        """Convert BCHW into B x regions x region_tokens x C."""
        b, c, _, _ = x.shape
        r = region_size
        return (
            x.view(b, c, region_h, r, region_w, r)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, region_h * region_w, r * r, c)
        )

    def _compress_kv_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress projected K/V maps into one weighted token per spatial group."""
        padded, _, _ = self._pad_to_multiple(x, self.group_size)
        _, _, hp, wp = padded.shape
        group_h, group_w = hp // self.group_size, wp // self.group_size

        k_map, v_map = self.kv(padded).chunk(2, dim=1)
        score_map = self.score(padded)
        k_tokens = self._to_groups(k_map, group_h, group_w)
        v_tokens = self._to_groups(v_map, group_h, group_w)
        score_tokens = self._to_groups(score_map, group_h, group_w).squeeze(-1)
        weights = score_tokens.softmax(dim=2).unsqueeze(-1)

        k_groups = (k_tokens * weights).sum(dim=2)
        v_groups = (v_tokens * weights).sum(dim=2)
        group_scores = score_tokens.mean(dim=2)
        return k_groups, v_groups, group_scores

    def _format_full_q(self, q_map: torch.Tensor) -> torch.Tensor:
        """Convert full-resolution Q map to B x heads x tokens x head_dim."""
        b, c, h, w = q_map.shape
        q = q_map.flatten(2).transpose(1, 2)
        return q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _format_selected_kv(self, selected: torch.Tensor) -> torch.Tensor:
        """Convert B x tokens x C selected K/V tokens to B x heads x tokens x head_dim."""
        b, n, _ = selected.shape
        return selected.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


class TopKGlobalGroupKVAttention(_TopKGroupKVAttentionBase):
    """Full-query attention over a global top-k set of compressed K/V groups."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, group_size: int = 4, topk: int = 100):
        """Initialize global top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select one top-k group set per image and attend all query tokens to it."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        q = self._format_full_q(self.q(x))
        k_groups, v_groups, group_scores = self._compress_kv_groups(x)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = group_scores.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]
        k = self._format_selected_kv(k_groups[batch_idx, route_idx])
        v = self._format_selected_kv(v_groups[batch_idx, route_idx])

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_bn(self.proj(out))


class TopKAdaptiveGroupKVAttention(_TopKGroupKVAttentionBase):
    """Region-wise query attention over adaptive top-k compressed K/V groups."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        group_size: int = 4,
        query_region_size: int = 10,
        topk: int = 8,
    ):
        """Initialize adaptive top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.query_region_size = max(1, int(query_region_size))
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select top-k K/V groups per query region, then attend region tokens to them."""
        x = self.input_proj(x)
        b, c, _, _ = x.shape
        q_map = self.q(x)
        q_padded, orig_h, orig_w = self._pad_to_multiple(q_map, self.query_region_size)
        _, _, hp, wp = q_padded.shape
        region_h, region_w = hp // self.query_region_size, wp // self.query_region_size
        num_regions = region_h * region_w
        tokens_per_region = self.query_region_size * self.query_region_size

        q_regions = self._to_regions(q_padded, self.query_region_size, region_h, region_w)
        q_repr = q_regions.mean(dim=2)
        k_groups, v_groups, _ = self._compress_kv_groups(x)
        affinity = (q_repr @ k_groups.transpose(-2, -1)) * (c**-0.5)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]

        outputs = []
        for region_idx in range(num_regions):
            selected = route_idx[:, region_idx]
            q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
            q_tokens = q_tokens.permute(0, 2, 1, 3)
            k_tokens = self._format_selected_kv(k_groups[batch_idx, selected])
            v_tokens = self._format_selected_kv(v_groups[batch_idx, selected])
            with torch.cuda.amp.autocast(enabled=False):
                attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
            out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
            outputs.append(out)

        out_regions = torch.stack(outputs, dim=1)
        r = self.query_region_size
        out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
        return x + self.proj_bn(self.proj(out))
