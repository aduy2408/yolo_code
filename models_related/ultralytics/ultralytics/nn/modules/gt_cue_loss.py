"""GT-Guided Cue Preservation (#5) Auxiliary Supervision Modules.

Computes object-local descriptor targets T = [ΔCb, ΔCr, ΔY, V] from raw RGB image
and GT bounding boxes, and supervises intermediate P2 representations via a light auxiliary head.
Inference path remains 100% untouched: F -> Detect.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_raw_color_and_detail(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Cb, Cr, Y, and HighPass Y from input RGB image (B, 3, H_img, W_img) in range [0, 1]."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    cb = -0.3374 * r - 0.6626 * g + b
    cr = r - 0.8374 * g - 0.1626 * b
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Gaussian blur 5x5 for high-pass detail: Y - Blur5(Y)
    blur_kernel = torch.tensor(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
        dtype=img.dtype,
        device=img.device,
    ) / 256.0
    blur_kernel = blur_kernel.view(1, 1, 5, 5)
    y_blur = F.conv2d(y, blur_kernel, padding=2)
    v_highpass = torch.abs(y - y_blur)

    return cb, cr, y, v_highpass


class GTCuePreservationHead(nn.Module):
    """Light auxiliary decoder D_cue(F): 32ch -> 16ch -> 4ch for GT cue target prediction."""

    def __init__(self, c1: int = 32, c2: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(16, c2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking F (B, 32, H, W) -> T_hat (B, 4, H, W)."""
        return self.conv2(self.dwconv(self.conv1(x)))


def build_gt_cue_targets(
    img: torch.Tensor,
    targets: torch.Tensor,
    feat_shape: tuple[int, int] = (128, 128),
    p2_stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build GT target tensor T (B, 4, H, W) and pos_mask (B, H, W) for GT-positive P2 grid cells.

    Targets format from Ultralytics batch:
        targets: (N_gt, 6) tensor of [img_idx, class_id, x_norm, y_norm, w_norm, h_norm]
    """
    b, _, h_img, w_img = img.shape
    h_feat, w_feat = feat_shape
    device = img.device
    dtype = img.dtype

    cb, cr, y, v_highpass = compute_raw_color_and_detail(img)

    target_T = torch.zeros((b, 4, h_feat, w_feat), dtype=dtype, device=device)
    pos_mask = torch.zeros((b, h_feat, w_feat), dtype=torch.bool, device=device)

    if targets is None or targets.numel() == 0:
        return target_T, pos_mask

    # Process each batch item
    for batch_idx in range(b):
        gt_mask = targets[:, 0] == batch_idx
        b_targets = targets[gt_mask]
        if b_targets.numel() == 0:
            continue

        for target in b_targets:
            _, _, x_c, y_c, w_b, h_b = target.tolist()

            # Bounding box in image pixel coordinates
            x1 = max(0, int((x_c - w_b / 2) * w_img))
            y1 = max(0, int((y_c - h_b / 2) * h_img))
            x2 = min(w_img, int((x_c + w_b / 2) * w_img))
            y2 = min(h_img, int((y_c + h_b / 2) * h_img))

            if x2 <= x1 or y2 <= y1:
                continue

            # Neighborhood expansion (1 P2 cell = 4 image pixels)
            exp = p2_stride
            x1_near, y1_near = max(0, x1 - exp), max(0, y1 - exp)
            x2_near, y2_near = min(w_img, x2 + exp), min(h_img, y2 + exp)

            # Extract inner vs neighborhood means
            mean_cb_in = cb[batch_idx, 0, y1:y2, x1:x2].mean()
            mean_cr_in = cr[batch_idx, 0, y1:y2, x1:x2].mean()
            mean_y_in = y[batch_idx, 0, y1:y2, x1:x2].mean()
            mean_v_in = v_highpass[batch_idx, 0, y1:y2, x1:x2].mean()

            mean_cb_near = cb[batch_idx, 0, y1_near:y2_near, x1_near:x2_near].mean()
            mean_cr_near = cr[batch_idx, 0, y1_near:y2_near, x1_near:x2_near].mean()
            mean_y_near = y[batch_idx, 0, y1_near:y2_near, x1_near:x2_near].mean()

            delta_cb = mean_cb_in - mean_cb_near
            delta_cr = mean_cr_in - mean_cr_near
            delta_y = mean_y_in - mean_y_near

            t_vec = torch.tensor([delta_cb, delta_cr, delta_y, mean_v_in], dtype=dtype, device=device)

            # Map GT bbox to P2 feature map grid cells
            gx1 = max(0, int(x1 / p2_stride))
            gy1 = max(0, int(y1 / p2_stride))
            gx2 = min(w_feat, int((x2 + p2_stride - 1) / p2_stride))
            gy2 = min(h_feat, int((y2 + p2_stride - 1) / p2_stride))

            if gx2 > gx1 and gy2 > gy1:
                target_T[batch_idx, :, gy1:gy2, gx1:gx2] = t_vec.view(4, 1, 1)
                pos_mask[batch_idx, gy1:gy2, gx1:gx2] = True

    return target_T, pos_mask


def compute_gt_cue_loss(
    pred_T: torch.Tensor,
    target_T: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute SmoothL1 loss strictly over GT-positive grid cells."""
    if not pos_mask.any():
        return pred_T.sum() * 0.0

    # Mask shapes: pred_T (B, 4, H, W), pos_mask (B, H, W)
    mask_4d = pos_mask.unsqueeze(1).expand_as(pred_T)
    pred_pos = pred_T[mask_4d]
    target_pos = target_T[mask_4d]

    loss = F.smooth_l1_loss(pred_pos, target_pos, reduction="mean")
    return loss
