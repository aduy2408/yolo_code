"""Project-owned detection heads built on clean upstream Detect."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import DWConv
from ultralytics.utils.tal import make_anchors

from .attention import KVCompressedAttention, KVCompressedTransformerEncoder
from .cbam import CBAM


P2_NUDFL_BINS = (
    0.0,
    0.35,
    0.70,
    1.05,
    1.40,
    1.80,
    2.30,
    2.90,
    3.60,
    4.50,
    5.60,
    6.90,
    8.40,
    10.20,
    12.40,
    15.00,
)


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


class P2NUDFLDetect(Detect):
    """Detect head carrying the historical P2 non-uniform DFL codebook.

    The project head preserves the codebook as metadata for a future project
    loss adapter. Clean upstream DFL still performs the standard projection
    until that loss adapter is explicitly migrated.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("p2_dfl_bins", torch.tensor(P2_NUDFL_BINS), persistent=True)


class LocalInstanceDemixer(nn.Module):
    """Competitive 3x3 local evidence demixer for a small number of slots."""

    def __init__(self, channels: int, slots: int = 2, patch: int = 3) -> None:
        super().__init__()
        if slots != 2:
            raise ValueError("The v1 P2 demixer supports exactly two object slots")
        if patch != 3:
            raise ValueError("The v1 P2 demixer uses exactly a 3x3 neighborhood")
        self.slots = int(slots)
        self.radius = patch // 2
        self.query_count = patch * patch
        self.value = nn.Conv2d(channels, channels, 1)
        self.context = nn.Sequential(
            DWConv(channels, channels, 3),
            nn.Conv2d(channels, (self.slots + 1) * self.query_count, 1),
        )
        self.slot_embed = nn.Parameter(torch.zeros(self.slots, channels, 1, 1))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        value = self.value(feature)
        logits = self.context(feature).view(batch, self.slots + 1, self.query_count, height, width)
        responsibilities = logits.softmax(dim=1)
        padded = F.pad(value, (self.radius,) * 4)
        mixed = torch.zeros(
            batch, self.slots, channels, height, width, device=feature.device, dtype=feature.dtype
        )
        mass = torch.zeros(batch, self.slots, 1, height, width, device=feature.device, dtype=feature.dtype)
        query = 0
        for dy in range(2 * self.radius + 1):
            for dx in range(2 * self.radius + 1):
                value_q = padded[:, :, dy : dy + height, dx : dx + width]
                weight = responsibilities[:, : self.slots, query].unsqueeze(2)
                mixed = mixed + weight * value_q.unsqueeze(1)
                mass = mass + weight
                query += 1
        demixed = mixed / (mass + 1e-6)
        return (
            feature.unsqueeze(1)
            + self.slot_embed.unsqueeze(0)
            + self.gamma * (demixed - feature.unsqueeze(1))
        )


class P2SlotsDetect(Detect):
    """YOLO Detect with two P2 hypotheses and shared prediction heads.

    ``capacity`` adds only slot embeddings. ``demix`` additionally applies the
    competitive local demixer before both slots use the same P2 box/class head.
    P3 and P4 remain ordinary single-hypothesis Detect outputs.
    """

    def __init__(
        self,
        nc: int = 80,
        variant: str = "demix",
        slots: int = 2,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if end2end:
            raise ValueError("P2SlotsDetect supports standard one-to-many detection only")
        if len(ch) < 2:
            raise ValueError("P2SlotsDetect requires P2 plus at least one coarser feature level")
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self.p2_slot_count = int(slots)
        self.p2_variant = str(variant).lower()
        if self.p2_slot_count != 2:
            raise ValueError("P2SlotsDetect v1 supports exactly slots=2")
        if self.p2_variant not in {"capacity", "demix"}:
            raise ValueError("variant must be 'capacity' or 'demix'")
        self.p2_slot_embed = nn.Parameter(torch.zeros(self.p2_slot_count, ch[0], 1, 1))
        self.p2_demixer = LocalInstanceDemixer(ch[0], self.p2_slot_count) if self.p2_variant == "demix" else None

    def _p2_slots(self, feature: torch.Tensor) -> torch.Tensor:
        if self.p2_demixer is not None:
            return self.p2_demixer(feature)
        return feature.unsqueeze(1) + self.p2_slot_embed.unsqueeze(0)

    def _slot_predictions(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = feature.shape[0]
        slots = self._p2_slots(feature)
        boxes = torch.cat(
            [self.cv2[0](slots[:, index]).view(batch, 4 * self.reg_max, -1) for index in range(self.p2_slot_count)],
            dim=-1,
        )
        scores = torch.cat(
            [self.cv3[0](slots[:, index]).view(batch, self.nc, -1) for index in range(self.p2_slot_count)],
            dim=-1,
        )
        return boxes, scores

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        batch = x[0].shape[0]
        p2_boxes, p2_scores = self._slot_predictions(x[0])
        boxes = [p2_boxes]
        scores = [p2_scores]
        for index in range(1, self.nl):
            boxes.append(box_head[index](x[index]).view(batch, 4 * self.reg_max, -1))
            scores.append(cls_head[index](x[index]).view(batch, self.nc, -1))
        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)
        p2_count = x[0].shape[-2] * x[0].shape[-1]
        return {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": x,
            "anchor_points": torch.cat((anchor_points[:p2_count].repeat(self.p2_slot_count, 1), anchor_points[p2_count:]), dim=0),
            "stride_tensor": torch.cat((stride_tensor[:p2_count].repeat(self.p2_slot_count, 1), stride_tensor[p2_count:]), dim=0),
            "p2_slot_count": self.p2_slot_count,
            "p2_base_count": p2_count,
        }

    def _get_decode_boxes(self, x):
        if "anchor_points" not in x:
            return super()._get_decode_boxes(x)
        boxes = self.decode_bboxes(self.dfl(x["boxes"]), x["anchor_points"].t().unsqueeze(0))
        return boxes * x["stride_tensor"].t().unsqueeze(0)
