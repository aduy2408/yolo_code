"""Project-owned detection heads built on clean upstream Detect."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect

from .attention import KVCompressedAttention, KVCompressedTransformerEncoder
from .cbam import CBAM


class DetectClsAttention(Detect):
    """YOLO Detect with an optional P2 classification-feature attention block.

    This migrates the useful ``cbam``, ``kvca``, and ``kvca_block`` variants
    without importing the legacy detection head or changing upstream Detect.
    """

    def __init__(
        self,
        nc: int = 80,
        attn_type: str = "cbam",
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.attn_type = str(attn_type).lower()
        c_p2 = ch[0]
        if self.attn_type == "cbam":
            self.attn = CBAM(c_p2)
        elif self.attn_type == "kvca":
            self.attn = KVCompressedTransformerEncoder(c_p2, c_p2, num_heads=4, sr_ratio=8, mode="dwconv")
        elif self.attn_type == "kvca_block":
            self.attn = KVCompressedAttention(c_p2, c_p2, num_heads=4, sr_ratio=8, mode="group_weight")
        else:
            raise ValueError(
                f"Unsupported project DetectClsAttention type {self.attn_type!r}; "
                "supported values are cbam, kvca, kvca_block"
            )

    def forward(self, x: list[torch.Tensor]):
        """Apply attention to P2 classification features and run Detect."""
        cls_x = [self.attn(x[0]), *x[1:]]
        preds = self.forward_head(x, **self.one2many)
        # Rebuild only the classification branch with attended P2 features.
        preds["scores"] = torch.cat(
            [self.one2many["cls_head"][i](cls_x[i]).view(x[0].shape[0], self.nc, -1) for i in range(self.nl)], dim=-1
        )
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)
