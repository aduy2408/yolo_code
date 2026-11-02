"""All auxiliary supervision and channel specialization modules for experiments #3, #4, #5, #6.

#3 DedicatedCueSlots        F*=[F24,B4,C2,E1,H1] - pure concat, no learned mixing
#4 DetachedResidualFusion   F*=[F24,E8] where E8=B8+φ(B8, sg(C4-D(F)))
#5 GTCuePreservationHead    F=Detect, auxiliary D_cue(F)->T hat, loss only during training
#6a SplitChannelDetect      F*=[S24,D8], no auxiliary supervision (ablation control)
#6b GTChannelSpecialization F*=[S24,D8], D8 receives GT cue supervision
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_blur5(x: torch.Tensor) -> torch.Tensor:
    """Apply 5x5 Gaussian blur to single-channel tensor (B, 1, H, W)."""
    k = torch.tensor(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
        dtype=x.dtype, device=x.device,
    ) / 256.0
    return F.conv2d(x, k.view(1, 1, 5, 5), padding=2)


def compute_raw_cue_channels(img: torch.Tensor) -> tuple:
    """Compute (Cb, Cr, Y, E, H) from img (B,3,H,W) in [0,1].
    Cb, Cr: chroma (2ch)
    Y: luminance (1ch)
    E: gradient magnitude |Gx|+|Gy| (1ch)
    H: high-pass Y - Blur5(Y) (1ch)
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    cb = -0.3374 * r - 0.6626 * g + b
    cr = r - 0.8374 * g - 0.1626 * b
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Sobel gradient magnitude (L1 approximation)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    gx = F.conv2d(y, sobel_x, padding=1)
    gy = F.conv2d(y, sobel_y, padding=1)
    e = torch.abs(gx) + torch.abs(gy)

    # High-pass
    h = y - _gaussian_blur5(y)

    return cb, cr, y, e, h


def _pool_to_feat(x: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """AvgPool stride=4 to match P2 feature map grid."""
    return F.avg_pool2d(x, kernel_size=stride, stride=stride, padding=0)


def build_gt_cue_targets(
    img: torch.Tensor,
    targets: torch.Tensor,
    feat_shape: tuple[int, int] = (128, 128),
    p2_stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build GT target tensor T=(B,4,H,W) and pos_mask=(B,H,W) over GT-positive P2 cells.

    T = [ΔCb, ΔCr, ΔY, V] where Δ = inner_mean - near_mean, V = inner_mean(|highpass|).
    targets: (N,6) [img_idx, cls, cx_norm, cy_norm, w_norm, h_norm]
    """
    b, _, h_img, w_img = img.shape
    h_feat, w_feat = feat_shape
    device, dtype = img.device, img.dtype

    cb, cr, y, _, h = compute_raw_cue_channels(img)
    v_hp = torch.abs(h)

    target_T = torch.zeros((b, 4, h_feat, w_feat), dtype=dtype, device=device)
    pos_mask = torch.zeros((b, h_feat, w_feat), dtype=torch.bool, device=device)

    if targets is None or targets.numel() == 0:
        return target_T, pos_mask

    for batch_idx in range(b):
        bt = targets[targets[:, 0] == batch_idx]
        if bt.numel() == 0:
            continue
        for t in bt:
            _, _, x_c, y_c, w_b, h_b = t.tolist()
            x1, y1 = max(0, int((x_c - w_b / 2) * w_img)), max(0, int((y_c - h_b / 2) * h_img))
            x2, y2 = min(w_img, int((x_c + w_b / 2) * w_img)), min(h_img, int((y_c + h_b / 2) * h_img))
            if x2 <= x1 or y2 <= y1:
                continue
            ex = p2_stride
            xn1, yn1 = max(0, x1 - ex), max(0, y1 - ex)
            xn2, yn2 = min(w_img, x2 + ex), min(h_img, y2 + ex)

            def _m(t): return t[batch_idx, 0, y1:y2, x1:x2].mean()
            def _mn(t): return t[batch_idx, 0, yn1:yn2, xn1:xn2].mean()

            t_vec = torch.tensor([
                _m(cb) - _mn(cb), _m(cr) - _mn(cr),
                _m(y) - _mn(y), _m(v_hp),
            ], dtype=dtype, device=device)

            gx1, gy1 = max(0, x1 // p2_stride), max(0, y1 // p2_stride)
            gx2 = min(w_feat, (x2 + p2_stride - 1) // p2_stride)
            gy2 = min(h_feat, (y2 + p2_stride - 1) // p2_stride)
            if gx2 > gx1 and gy2 > gy1:
                target_T[batch_idx, :, gy1:gy2, gx1:gx2] = t_vec.view(4, 1, 1)
                pos_mask[batch_idx, gy1:gy2, gx1:gx2] = True

    return target_T, pos_mask


def compute_gt_cue_loss(pred_T, target_T, pos_mask):
    """SmoothL1 over GT-positive cells only."""
    if not pos_mask.any():
        return pred_T.sum() * 0.0
    m = pos_mask.unsqueeze(1).expand_as(pred_T)
    return F.smooth_l1_loss(pred_T[m], target_T[m], reduction="mean")


def _build_targets_from_batch(batch, device):
    batch_idx = batch.get("batch_idx")
    bboxes = batch.get("bboxes")
    if batch_idx is None or bboxes is None or bboxes.numel() == 0:
        return None
    cls = batch.get("cls", torch.zeros(batch_idx.shape[0], 1, device=device))
    if cls.ndim == 2:
        cls = cls[:, 0]
    return torch.stack([batch_idx.float(), cls.float(),
                        bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# #3 Dedicated Cue Slots: F* = [F24, B4, C2, E1, H1] — zero learned mixing
# ─────────────────────────────────────────────────────────────────────────────

class DedicatedCueSlots(nn.Module):
    """#3: F* = [PF(F)_24, PB(B)_4, norm(Cb)_1, norm(Cr)_1, norm(E)_1, norm(H)_1].

    No learned mixing between modalities. Only two 1×1 projections for F and B.
    All raw cue channels are simply AvgPooled and channel-normalized.
    """

    def __init__(self, c_f: int = 24, c_b: int = 4) -> None:
        super().__init__()
        self.proj_f = nn.Sequential(
            nn.Conv2d(32, c_f, 1, bias=False), nn.BatchNorm2d(c_f), nn.SiLU(inplace=True)
        )
        self.proj_b = nn.Sequential(
            nn.Conv2d(32, c_b, 1, bias=False), nn.BatchNorm2d(c_b), nn.SiLU(inplace=True)
        )
        # Learnable per-channel affine for raw cues (no mixing)
        # c2=2, e1=1, h1=1 → 4 channels total
        self.affine_scale = nn.Parameter(torch.ones(4, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(4, 1, 1))

    def forward(self, x: list[torch.Tensor], img0: torch.Tensor) -> torch.Tensor:
        f, b = x[0], x[1]
        cb, cr, _, e, h = compute_raw_cue_channels(img0)
        stride = img0.shape[2] // f.shape[2]
        cues = torch.cat([
            _pool_to_feat(cb, stride),
            _pool_to_feat(cr, stride),
            _pool_to_feat(e, stride),
            _pool_to_feat(h, stride),
        ], dim=1)  # (B, 4, H, W)
        cues = cues * self.affine_scale + self.affine_bias
        return torch.cat([self.proj_f(f), self.proj_b(b), cues], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# #4 Detached Residual Fusion: F* = [F24, E8] where E8 = B8 + φ(B8, sg(C4-D(F)))
# ─────────────────────────────────────────────────────────────────────────────

class _LightDecoder(nn.Module):
    """D(F): F32 → 1×1(32→16) → DWConv3×3 → 1×1(16→4)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.dw = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.conv2 = nn.Conv2d(16, 4, 1)
        # Zero-init final conv → D(F)=0 at start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.dw(self.conv1(f)))


class _FormationBlock(nn.Module):
    """φ(B8, R4) → E8: 1×1(12→16) → DWConv3×3 → 1×1(16→8), zero-init last conv."""

    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(c_in, 16, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.dw = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.conv2 = nn.Conv2d(16, 8, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.dw(self.conv1(x)))


class DetachedResidualFusion(nn.Module):
    """#4: F* = [F24, E8] where E8 = B8 + φ(B8, sg(C4 - D(F))).

    D(F) learns to predict C4 from F via separate L1 reconstruction loss (recon_gain * L1(D(F), C4)).
    The residual R = sg(C4 - D(F)) is stop-graded so detection loss cannot degrade D.
    D(F) only learns from reconstruction objective.
    """

    def __init__(self, recon_gain: float = 0.5) -> None:
        super().__init__()
        self.proj_f = nn.Sequential(nn.Conv2d(32, 24, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(inplace=True))
        self.proj_b = nn.Sequential(nn.Conv2d(32, 8, 1, bias=False), nn.BatchNorm2d(8), nn.SiLU(inplace=True))
        self.decoder = _LightDecoder()          # D(F) → C4_hat
        self.phi = _FormationBlock(8 + 4)       # φ([B8, R4]) → ΔE8
        self.recon_gain = recon_gain
        self._last_recon_loss: torch.Tensor | None = None

    def forward(self, x: list[torch.Tensor], img0: torch.Tensor) -> torch.Tensor:
        f, b = x[0], x[1]
        stride = img0.shape[2] // f.shape[2]
        cb, cr, _, e, h = compute_raw_cue_channels(img0)
        c4 = torch.cat([
            _pool_to_feat(cb, stride), _pool_to_feat(cr, stride),
            _pool_to_feat(e, stride), _pool_to_feat(h, stride),
        ], dim=1)  # (B, 4, H, W) — C4 at stride 4

        c4_hat = self.decoder(f)               # predicted C4 from F
        if self.training:
            recon = F.l1_loss(c4_hat, c4.detach())
            self._last_recon_loss = recon * self.recon_gain

        r = (c4 - c4_hat).detach()            # sg(C4 - D(F)) — stop gradient
        b8 = self.proj_b(b)
        e8 = b8 + self.phi(torch.cat([b8, r], dim=1))

        return torch.cat([self.proj_f(f), e8], dim=1)

    def auxiliary_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return accumulated reconstruction loss for DetectionModel.loss() dispatch."""
        if self._last_recon_loss is None:
            return self.decoder.conv2.weight.sum() * 0.0, {}
        loss = self._last_recon_loss
        self._last_recon_loss = None
        return loss, {"recon_loss": float(loss.detach())}


# ─────────────────────────────────────────────────────────────────────────────
# #5 GT Cue Preservation Head (unchanged from original gt_cue_loss.py)
# Also defined here so this file is the single source
# ─────────────────────────────────────────────────────────────────────────────

class GTCuePreservationHead(nn.Module):
    """#5: Auxiliary D_cue(F): 32→16→4 predicting GT object-local descriptor T.

    Inference path unchanged: F→Detect. Only used during training via auxiliary_loss().
    """

    def __init__(self, c1: int = 32, c2: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(c1, 16, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.dwconv = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.SiLU(inplace=True))
        self.conv2 = nn.Conv2d(16, c2, 1)
        self.last_pred: torch.Tensor | None = None
        self.cue_gain: float = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.conv2(self.dwconv(self.conv1(x)))
        if self.training:
            self.last_pred = pred
        return pred

    def auxiliary_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self.last_pred is None:
            return self.conv2.weight.sum() * 0.0, {}
        pred_T = self.last_pred
        self.last_pred = None
        img = batch["img"]
        targets = _build_targets_from_batch(batch, img.device)
        if targets is None:
            return pred_T.sum() * 0.0, {"gt_cue_loss": 0.0}
        h_feat, w_feat = pred_T.shape[2], pred_T.shape[3]
        target_T, pos_mask = build_gt_cue_targets(img, targets, feat_shape=(h_feat, w_feat), p2_stride=img.shape[2] // h_feat)
        loss = compute_gt_cue_loss(pred_T, target_T, pos_mask)
        n_pos = int(pos_mask.sum().item())
        return loss * self.cue_gain, {"gt_cue_loss": float(loss.detach()), "gt_cue_n_pos": n_pos}


# ─────────────────────────────────────────────────────────────────────────────
# #6a Split-Channel Detect (control — no aux loss)
# ─────────────────────────────────────────────────────────────────────────────

class SplitChannelDetect(nn.Module):
    """#6a: F* = [Ps(F)_24, Pd(F)_8] — explicit semantic/detail split, no auxiliary supervision.

    Control ablation for #6b: tests whether the split itself (not the cue supervision) drives gain.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj_s = nn.Sequential(nn.Conv2d(32, 24, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(inplace=True))
        self.proj_d = nn.Sequential(nn.Conv2d(32, 8, 1, bias=False), nn.BatchNorm2d(8), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.proj_s(x), self.proj_d(x)], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# #6b GT Channel Specialization
# ─────────────────────────────────────────────────────────────────────────────

class GTChannelSpecialization(nn.Module):
    """#6b: F* = [Ps(F)_24, Pd(F)_8] with D8 receiving GT cue auxiliary supervision.

    Decoder 1×1(8→4) predicts T=[chroma CS, luminance CS, boundary, texture] from D8.
    Paired with SplitChannelDetect (6a) as ablation: 6b - 6a isolates D8 cue supervision effect.
    """

    def __init__(self, cue_gain: float = 0.1) -> None:
        super().__init__()
        self.proj_s = nn.Sequential(nn.Conv2d(32, 24, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(inplace=True))
        self.proj_d = nn.Sequential(nn.Conv2d(32, 8, 1, bias=False), nn.BatchNorm2d(8), nn.SiLU(inplace=True))
        # Auxiliary decoder on D8 only
        self.decoder = nn.Conv2d(8, 4, 1)
        self.cue_gain = cue_gain
        self._last_d8: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.proj_s(x)
        d = self.proj_d(x)
        if self.training:
            self._last_d8 = d
        return torch.cat([s, d], dim=1)

    def auxiliary_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self._last_d8 is None:
            return self.decoder.weight.sum() * 0.0, {}
        d8 = self._last_d8
        self._last_d8 = None
        img = batch["img"]
        targets = _build_targets_from_batch(batch, img.device)
        if targets is None:
            return d8.sum() * 0.0, {"spec_cue_loss": 0.0}
        pred_T = self.decoder(d8)
        h_feat, w_feat = pred_T.shape[2], pred_T.shape[3]
        target_T, pos_mask = build_gt_cue_targets(img, targets, feat_shape=(h_feat, w_feat), p2_stride=img.shape[2] // h_feat)
        loss = compute_gt_cue_loss(pred_T, target_T, pos_mask)
        n_pos = int(pos_mask.sum().item())
        return loss * self.cue_gain, {"spec_cue_loss": float(loss.detach()), "spec_cue_n_pos": n_pos}
