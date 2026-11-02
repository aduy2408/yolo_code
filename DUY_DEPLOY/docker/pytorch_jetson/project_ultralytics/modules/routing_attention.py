"""Project-specific bi-level routed attention.

Extracted from the legacy fork and implemented against clean upstream Conv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


def _choose_attention_heads(channels: int, requested_heads: int) -> int:
    """Pick a valid attention head count that divides channels."""
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1

class BiLevelRoutingAttention(nn.Module):
    """NCHW bi-level routing attention adapted for YOLO feature maps."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        n_win: int = 7,
        topk: int = 4,
        side_dwconv: int = 3,
    ):
        """Initialize bi-level routing attention."""
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = c2**-0.5
        self.n_win = max(1, int(n_win))
        self.topk = max(1, int(topk))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.qkv_linear = nn.Conv2d(c2, c2 * 3, 1)
        self.lepe = (
            nn.Conv2d(c2, c2, side_dwconv, stride=1, padding=side_dwconv // 2, groups=c2)
            if side_dwconv > 0
            else None
        )
        self.output_linear = nn.Conv2d(c2, c2, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_region_size(x: torch.Tensor, region_size: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to multiples of region_size and return original H/W."""
        h, w = x.shape[-2:]
        pad_h = (region_size[0] - h % region_size[0]) % region_size[0]
        pad_w = (region_size[1] - w % region_size[1]) % region_size[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    @staticmethod
    def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
        """Convert BCHW to B x heads x regions x region_tokens x head_dim."""
        b, c, h, w = x.shape
        rh, rw = region_size
        region_h, region_w = h // rh, w // rw
        x = x.view(b, num_heads, c // num_heads, region_h, rh, region_w, rw)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        return x.view(b, num_heads, region_h * region_w, rh * rw, c // num_heads), region_h, region_w

    @staticmethod
    def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
        """Convert B x heads x regions x region_tokens x head_dim to BCHW."""
        b, num_heads, _, _, head_dim = x.shape
        rh, rw = region_size
        x = x.view(b, num_heads, region_h, region_w, rh, rw, head_dim)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        return x.view(b, num_heads * head_dim, region_h * rh, region_w * rw)

    def _regional_routing_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        region_graph: torch.Tensor,
        region_size: tuple[int, int],
    ) -> torch.Tensor:
        """Apply token attention from each query region to selected key/value regions."""
        query, orig_h, orig_w = self._pad_to_region_size(query, region_size)
        key, _, _ = self._pad_to_region_size(key, region_size)
        value, _, _ = self._pad_to_region_size(value, region_size)

        query, q_region_h, q_region_w = self._grid2seq(query, region_size, self.num_heads)
        key, _, _ = self._grid2seq(key, region_size, self.num_heads)
        value, _, _ = self._grid2seq(value, region_size, self.num_heads)

        b, num_heads, q_regions, topk = region_graph.shape
        _, _, kv_regions, kv_region_tokens, head_dim = key.shape
        index = region_graph.view(b, num_heads, q_regions, topk, 1, 1).expand(
            -1, -1, -1, -1, kv_region_tokens, head_dim
        )
        key_g = torch.gather(
            key.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )
        value_g = torch.gather(
            value.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )

        with torch.cuda.amp.autocast(enabled=False):
            attn = (query.float() * self.scale) @ key_g.flatten(-3, -2).float().transpose(-1, -2)
            attn = attn.softmax(dim=-1)
        out = attn.to(value_g.dtype) @ value_g.flatten(-3, -2)
        out = self._seq2grid(out, q_region_h, q_region_w, region_size)
        return out[:, :, :orig_h, :orig_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route query regions to top-k key/value regions and apply token attention."""
        x = self.input_proj(x)
        _, c, h, w = x.shape
        region_size = (max(1, h // self.n_win), max(1, w // self.n_win))

        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=1)

        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)
        k_r = k_r.flatten(2, 3)
        affinity = q_r @ k_r
        route_count = min(self.topk, k_r.shape[-1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        route_idx = route_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        out = self._regional_routing_attention(q, k, v, route_idx, region_size)
        lepe = self.lepe(v) if self.lepe is not None else torch.zeros_like(v)
        out = out + lepe
        return x + self.output_bn(self.output_linear(out.reshape(-1, c, h, w)))
