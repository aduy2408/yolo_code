#!/usr/bin/env python3
import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kornia.morphology import dilation, erosion
from segmentation_related.modules import BottleneckGFT, GFT, M2Skip, M3Skip, Mlp, OverlapPatchEmbed, PSUp, PyramidPool, ReparamConv
from natten import NeighborhoodAttention2D


class VarroaBBoxMaskDataset(Dataset):
    """Varroa dataset with rectangular segmentation masks generated from bbox labels."""

    def __init__(self, root=".", split="train", transform=None, size_divisor=16):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.size_divisor = size_divisor
        split_root = self._resolve_split_root(self.root, split)
        self.image_dir = split_root / "videos"
        self.label_dir = split_root / "labels"
        self.images = sorted(self.image_dir.rglob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No PNG images found under {self.image_dir}")

    @staticmethod
    def _resolve_split_root(root, split):
        candidates = [
            root / split,
            root / split / split,
            root,
        ]
        for candidate in candidates:
            if (candidate / "videos").is_dir() and (candidate / "labels").is_dir():
                return candidate

        checked = "\n".join(str(candidate / "videos") for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find videos/labels for split '{split}'. Checked:\n{checked}"
        )

    def __len__(self):
        return len(self.images)

    def _label_path(self, image_path):
        rel = image_path.relative_to(self.image_dir)
        return (self.label_dir / rel).with_suffix(".txt")

    @staticmethod
    def _read_boxes(label_path):
        if not label_path.exists():
            return []

        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        boxes = []
        for line in lines[1:]:
            values = [float(x) for x in line.replace(",", " ").split()]
            for i in range(0, len(values) - 3, 4):
                boxes.append(values[i:i + 4])
        return boxes

    @staticmethod
    def _boxes_to_mask(boxes, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            left = int(np.floor(min(x1, x2)))
            right = int(np.ceil(max(x1, x2)))
            top = int(np.floor(min(y1, y2)))
            bottom = int(np.ceil(max(y1, y2)))

            left = max(0, min(left, width - 1))
            right = max(0, min(right, width))
            top = max(0, min(top, height - 1))
            bottom = max(0, min(bottom, height))

            if right > left and bottom > top:
                mask[top:bottom, left:right] = 1
        return mask

    def _pad_to_divisor(self, image, mask):
        if not self.size_divisor:
            return image, mask

        height, width = mask.shape[:2]
        padded_height = int(np.ceil(height / self.size_divisor) * self.size_divisor)
        padded_width = int(np.ceil(width / self.size_divisor) * self.size_divisor)
        pad_bottom = padded_height - height
        pad_right = padded_width - width

        if pad_bottom == 0 and pad_right == 0:
            return image, mask

        image = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        mask = cv2.copyMakeBorder(
            mask,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        return image, mask

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        boxes = self._read_boxes(self._label_path(image_path))
        mask = self._boxes_to_mask(boxes, width=width, height=height)
        image, mask = self._pad_to_divisor(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        return image, mask


def get_transforms(input_height, input_width, train=True, resize_mode="pad"):
    if resize_mode == "stretch":
        transforms = [A.Resize(input_height, input_width)]
    elif resize_mode == "pad":
        transforms = [
            A.PadIfNeeded(
                min_height=input_height,
                min_width=input_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            )
        ]
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    if train:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.25),
        ]
    transforms += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


class NeighborhoodTransformer(nn.Module):
    def __init__(self, patchsize, img_size, in_channels, out_channel, stride, num_heads=8):
        super().__init__()
        assert out_channel % num_heads == 0, f"embed_dim {out_channel} not divisible by heads {num_heads}"
        self.patchembedding = OverlapPatchEmbed(patchsize, img_size, in_channels, out_channel, stride, "nat")
        self.norm1 = nn.LayerNorm(out_channel)
        self.att1 = NeighborhoodAttention2D(embed_dim=out_channel, num_heads=num_heads, kernel_size=3)
        self.norm2 = nn.LayerNorm(out_channel)
        self.mlp = Mlp(out_channel, 2 * out_channel, out_channel)

    def forward(self, x):
        x_embedding = self.patchembedding(x)
        x = self.norm1(x_embedding)
        att = self.att1(x) + x_embedding
        x = self.mlp(self.norm2(att)) + att
        return x.permute(0, 3, 1, 2).contiguous()


class PartialNeighborhoodTransformer(nn.Module):
    """Apply NATTEN to a channel subset, then fuse it with a cheap identity branch."""

    def __init__(self, channels, attn_ratio=0.5, num_heads=3, kernel_size=3):
        super().__init__()
        if not 0.0 < attn_ratio <= 1.0:
            raise ValueError(f"attn_ratio must be in (0, 1], got {attn_ratio}")

        desired_channels = int(round(channels * attn_ratio))
        attn_channels = max(8, int(round(desired_channels / 8)) * 8)
        attn_channels = min(channels, attn_channels)
        attn_channels = max(8, attn_channels)
        chosen_heads = max(1, attn_channels // 8)
        chosen_heads = min(num_heads, chosen_heads)
        while attn_channels % chosen_heads != 0 or (attn_channels // chosen_heads) % 8 != 0:
            chosen_heads -= 1
            if chosen_heads < 1:
                raise ValueError(
                    f"Cannot build partial NATTEN for channels={channels}, attn_channels={attn_channels}"
                )

        self.attn_channels = attn_channels
        self.skip_channels = channels - attn_channels
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.LayerNorm(attn_channels)
        self.attn = NeighborhoodAttention2D(embed_dim=attn_channels, num_heads=chosen_heads, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(attn_channels)
        self.mlp = Mlp(attn_channels, 2 * attn_channels, attn_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.pre(x)
        x_attn, x_skip = torch.split(x, [self.attn_channels, self.skip_channels], dim=1)

        x_attn = x_attn.permute(0, 2, 3, 1).contiguous()
        attn_input = self.norm1(x_attn)
        attn = self.attn(attn_input) + x_attn
        x_attn = self.mlp(self.norm2(attn)) + attn
        x_attn = x_attn.permute(0, 3, 1, 2).contiguous()

        if self.skip_channels:
            x = torch.cat([x_attn, x_skip], dim=1)
        else:
            x = x_attn
        return self.fuse(x)


def make_skip_attention(kind, channels, num_heads, ratio):
    if kind == "full":
        return NeighborhoodTransformer(3, 0, channels, channels, 1, num_heads)
    if kind == "partial":
        return PartialNeighborhoodTransformer(channels, attn_ratio=ratio, num_heads=num_heads)
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported skip attention kind: {kind}")


def make_gft(kind, in_channels, out_channels, bottleneck_channels):
    try:
        bottleneck_channels = int(bottleneck_channels)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gft_bottleneck must be an integer channel count, got {bottleneck_channels!r}. "
            "Call LM_Net with keyword args, e.g. LM_Net(..., gft_kind='global', gft_bottleneck=128)."
        ) from exc

    if kind == "full":
        return GFT(3, 14, in_channels, 2, out_channels, 1, 12)
    if kind == "identity":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)
    if kind in ("global", "linear", "pooled"):
        num_heads = 4
        while bottleneck_channels % num_heads != 0:
            num_heads -= 1
            if num_heads < 1:
                raise ValueError(f"Invalid GFT bottleneck channels: {bottleneck_channels}")
        return BottleneckGFT(
            in_channels,
            out_channels,
            bottleneck_channels=bottleneck_channels,
            expand_ratios=2,
            num_heads=num_heads,
            attention=kind,
        )
    raise ValueError(f"Unsupported GFT kind: {kind}")


def make_upsample(kind, in_channels, out_channels):
    if kind == "psup":
        return PSUp(in_channels, out_channels)
    if kind == "bilinear_conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
    raise ValueError(f"Unsupported upsample kind: {kind}")


def make_reparam(in_channels, expand_channels, out_channels, se_kind):
    return ReparamConv(in_channels, expand_channels, out_channels, 5, 3, se_kind=se_kind)


class LM_Net(nn.Module):
    def __init__(
        self,
        channel,
        n_classes=2,
        filters=None,
        deep_supervision=False,
        skip_attention="partial",
        partial_ratios=None,
        gft_kind="global",
        gft_bottleneck=128,
        upsample_kind="psup",
        se_kind="sse",
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.filters = filters or [24, 24, 48, 96, 192]
        self.skip_attention = skip_attention
        self.gft_kind = gft_kind
        self.upsample_kind = upsample_kind
        self.se_kind = se_kind
        partial_ratios = partial_ratios or [0.5, 0.5, 0.5, 0.5]
        f = self.filters

        self.conv1 = nn.Sequential(make_reparam(channel, f[1], f[0], se_kind), make_reparam(f[0], f[1], f[0], se_kind))
        self.down1 = nn.Conv2d(f[0], f[1], 3, 2, 1)
        self.conv2 = nn.Sequential(make_reparam(f[1], f[2], f[1], se_kind), make_reparam(f[1], f[2], f[1], se_kind))
        self.down2 = nn.Conv2d(f[1], f[2], 3, 2, 1)
        self.conv3 = nn.Sequential(make_reparam(f[2], f[3], f[2], se_kind), make_reparam(f[2], f[3], f[2], se_kind))
        self.down3 = nn.Conv2d(f[2], f[3], 3, 2, 1)
        self.conv4 = nn.Sequential(make_reparam(f[3], f[4], f[3], se_kind), make_reparam(f[3], f[4], f[3], se_kind))
        self.down4 = nn.Conv2d(f[3], f[4], 3, 2, 1)

        self.dconv1 = nn.Sequential(make_reparam(f[3], f[4], f[3], se_kind), make_reparam(f[3], f[4], f[3], se_kind))
        self.dconv2 = nn.Sequential(make_reparam(f[2], f[3], f[2], se_kind), make_reparam(f[2], f[3], f[2], se_kind))
        self.dconv3 = nn.Sequential(make_reparam(f[1], f[2], f[1], se_kind), make_reparam(f[1], f[2], f[1], se_kind))
        self.dconv4 = nn.Sequential(make_reparam(f[0], f[1], f[0], se_kind), make_reparam(f[0], f[1], f[0], se_kind))

        self.pyramidpool = PyramidPool()
        self.gft = make_gft(gft_kind, sum(f), f[4], gft_bottleneck)

        self.up1 = make_upsample(upsample_kind, f[4], f[3])
        self.up2 = make_upsample(upsample_kind, f[3], f[2])
        self.up3 = make_upsample(upsample_kind, f[2], f[1])
        self.up4 = make_upsample(upsample_kind, f[1], f[0])

        self.skip1 = M2Skip([f[2], f[3]], "bottom")
        self.skip2 = M3Skip([f[1], f[2], f[3]])
        self.skip3 = M3Skip([f[0], f[1], f[2]])
        self.skip4 = M2Skip([f[0], f[1]], "top")

        self.natt1 = make_skip_attention(skip_attention, f[3], 12, partial_ratios[0])
        self.natt2 = make_skip_attention(skip_attention, f[2], 6, partial_ratios[1])
        self.natt3 = make_skip_attention(skip_attention, f[1], 3, partial_ratios[2])
        self.natt4 = make_skip_attention(skip_attention, f[0], 3, partial_ratios[3])

        self.output_layer = nn.Conv2d(f[0], n_classes, 1)

    def structural_reparam(self):
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x_down4 = self.down4(x4)

        x5 = self.gft(self.pyramidpool(x1, x2, x3, x4, x_down4))

        x46 = self.natt1(self.skip1(x3, x4))
        x37 = self.natt2(self.skip2(x2, x3, x4))
        x28 = self.natt3(self.skip3(x1, x2, x3))
        x19 = self.natt4(self.skip4(x1, x2))

        x6 = self.dconv1(self.up1(x5) + x46)
        x7 = self.dconv2(self.up2(x6) + x37)
        x8 = self.dconv3(self.up3(x7) + x28)
        x9 = self.dconv4(self.up4(x8) + x19)
        return self.output_layer(x9)


def dice_loss(pred_probs, target_one_hot, smooth=1.0):
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    total = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    return 1 - ((2.0 * intersection + smooth) / (total + smooth)).mean()


def iou_loss(pred_probs, target_one_hot, smooth=1.0):
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    union = (pred_probs + target_one_hot - pred_probs * target_one_hot).sum(dim=(2, 3))
    return 1 - ((intersection + smooth) / (union + smooth)).mean()


def focal_loss(pred_logits, target_long, alpha=0.7, gamma=2.0):
    ce_loss = F.cross_entropy(pred_logits, target_long, reduction="none")
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()


class UltimateCombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        iou_weight=1.0,
        focal_weight=1.0,
        boundary_weight=0.0,
        connectivity_weight=0.0,
        class_weights=(0.2, 0.8),
        device="cuda",
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.iou_weight = float(iou_weight)
        self.focal_weight = float(focal_weight)
        self.boundary_weight = float(boundary_weight)
        self.connectivity_weight = float(connectivity_weight)
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
        self.kernel = torch.ones(5, 5, device=device)

    @torch.no_grad()
    def _distance_transform_bg(self, target_positive):
        bg = target_positive.cpu().numpy() == 0
        return torch.from_numpy(distance_transform_edt(bg)).float().to(target_positive.device)

    def get_relaxed_boundary_loss(self, pred_probs, target_one_hot):
        pred_positive = pred_probs[:, 1:2]
        target_positive = target_one_hot[:, 1:2]
        boundary_zone = dilation(target_positive, self.kernel)
        target_dist_map = self._distance_transform_bg(target_positive)
        return (torch.abs(pred_positive - target_positive) * target_dist_map * boundary_zone).mean()

    def get_soft_connectivity_loss(self, pred_probs):
        pred_positive = pred_probs[:, 1:2]
        return (pred_positive - erosion(pred_positive, self.kernel)).mean()

    def forward(self, pred_logits, target, return_components=False):
        target_long = target.long()
        pred_probs = F.softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target_long, num_classes=pred_logits.shape[1]).permute(0, 3, 1, 2).float()

        loss_ce = self.ce(pred_logits, target_long) if self.ce_weight else pred_logits.new_tensor(0.0)
        loss_dice = dice_loss(pred_probs, target_one_hot) if self.dice_weight else pred_logits.new_tensor(0.0)
        loss_iou = iou_loss(pred_probs, target_one_hot) if self.iou_weight else pred_logits.new_tensor(0.0)
        loss_focal = focal_loss(pred_logits, target_long) if self.focal_weight else pred_logits.new_tensor(0.0)
        loss_boundary = self.get_relaxed_boundary_loss(pred_probs, target_one_hot) if self.boundary_weight else pred_logits.new_tensor(0.0)
        loss_connectivity = self.get_soft_connectivity_loss(pred_probs) if self.connectivity_weight else pred_logits.new_tensor(0.0)

        total_loss = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.iou_weight * loss_iou
            + self.focal_weight * loss_focal
            + self.boundary_weight * loss_boundary
            + self.connectivity_weight * loss_connectivity
        )

        if not return_components:
            return total_loss

        return total_loss, {
            "total_loss": float(total_loss.detach().item()),
            "ce_loss": float(loss_ce.detach().item()),
            "dice_loss": float(loss_dice.detach().item()),
            "iou_loss": float(loss_iou.detach().item()),
            "focal_loss": float(loss_focal.detach().item()),
            "boundary_loss": float(loss_boundary.detach().item()),
            "connectivity_loss": float(loss_connectivity.detach().item()),
        }


class GlobalSegmentationMetrics:
    def __init__(self, device, positive_class=1):
        self.device = device
        self.positive_class = positive_class
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0.0, device=self.device)
        self.fp = torch.tensor(0.0, device=self.device)
        self.fn = torch.tensor(0.0, device=self.device)
        self.inter = torch.tensor(0.0, device=self.device)
        self.pred_sum = torch.tensor(0.0, device=self.device)
        self.target_sum = torch.tensor(0.0, device=self.device)
        self.union = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        p = (preds == self.positive_class).float()
        t = (targets == self.positive_class).float()
        tp = (p * t).sum()
        self.tp += tp
        self.fp += (p * (1 - t)).sum()
        self.fn += ((1 - p) * t).sum()
        self.inter += tp
        self.pred_sum += p.sum()
        self.target_sum += t.sum()
        self.union += p.sum() + t.sum() - tp

    def compute(self):
        eps = 1e-6
        return {
            "dice": ((2 * self.inter + eps) / (self.pred_sum + self.target_sum + eps)).item(),
            "iou": ((self.inter + eps) / (self.union + eps)).item(),
            "precision": ((self.tp + eps) / (self.tp + self.fp + eps)).item(),
            "recall": ((self.tp + eps) / (self.tp + self.fn + eps)).item(),
        }


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="best_lm_net_varroa.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_metrics = {}

    def __call__(self, val_loss, model, epoch, metrics):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.best_metrics = metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "metrics": metrics,
                },
                self.path,
            )
            print(f"Saved checkpoint: {self.path}")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def run_epoch(model, loader, criterion, optimizer, device, train=True, max_grad_norm=1.0, scaler=None, amp=False):
    model.train(train)
    metric = GlobalSegmentationMetrics(device)
    total_loss = 0.0
    total_components = None
    skipped_nonfinite = 0
    desc = "Training" if train else "Validation"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False, mininterval=0.5)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss, components = criterion(outputs, masks, return_components=True)

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"Warning: skipped non-finite loss batch ({desc}, skipped={skipped_nonfinite})")
                continue

            if train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            if total_components is None:
                total_components = {key: 0.0 for key in components}
            for key, value in components.items():
                total_components[key] += value

            total_loss += loss.item()
            metric.update(outputs.detach(), masks)
            current = metric.compute()
            pbar.set_postfix(dice=f"{current['dice']:.3f}", iou=f"{current['iou']:.3f}")

    num_batches = max(1, len(loader))
    effective_batches = max(1, num_batches - skipped_nonfinite)
    result = {"loss": total_loss / effective_batches, "skipped_nonfinite": skipped_nonfinite, **metric.compute()}
    if total_components:
        result.update({key: value / effective_batches for key, value in total_components.items()})
    return result


def append_metrics(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def parse_args():
    parser = argparse.ArgumentParser(description="Train LM_Net on Varroa bbox-derived rectangular masks.")
    parser.add_argument("--root", default=".", help="Dataset root containing train/val/test folders.")
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--size-divisor", type=int, default=16)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument(
        "--skip-attention",
        choices=("full", "partial", "identity"),
        default="partial",
        help="Attention used on LM_Net skip features.",
    )
    parser.add_argument(
        "--partial-ratios",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        metavar=("NATT1", "NATT2", "NATT3", "NATT4"),
        help="Channel ratios for partial skip attention from low-res to high-res skips.",
    )
    parser.add_argument(
        "--gft-kind",
        choices=("full", "global", "linear", "pooled", "identity"),
        default="global",
        help="GFT fusion block. full is the original 384-channel GFT; others use a bottleneck.",
    )
    parser.add_argument(
        "--gft-bottleneck",
        type=int,
        default=128,
        help="Channels used inside bottleneck GFT variants.",
    )
    parser.add_argument(
        "--upsample-kind",
        choices=("psup", "bilinear_conv"),
        default="psup",
        help="Decoder upsampling block. bilinear_conv matches the original bilinear upsample + 3x3 conv.",
    )
    parser.add_argument(
        "--se-kind",
        choices=("sse", "se"),
        default="sse",
        help="Attention gate inside ReparamConv. se matches original LM-Net; sse is the lightweight variant.",
    )
    parser.add_argument(
        "--filters",
        type=int,
        nargs=5,
        default=None,
        metavar=("F0", "F1", "F2", "F3", "F4"),
        help="Encoder/decoder channel list. Omit to keep the current default [24, 24, 48, 96, 192].",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("pad", "stretch"),
        default="pad",
        help="pad preserves image aspect ratio; stretch resizes to input-height/input-width.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint", default="best_lm_net_varroa.pt")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--connectivity-weight", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = VarroaBBoxMaskDataset(
        args.root,
        "train",
        get_transforms(args.input_height, args.input_width, train=True, resize_mode=args.resize_mode),
        size_divisor=args.size_divisor,
    )
    val_dataset = VarroaBBoxMaskDataset(
        args.root,
        "val",
        get_transforms(args.input_height, args.input_width, train=False, resize_mode=args.resize_mode),
        size_divisor=args.size_divisor,
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = LM_Net(
        channel=3,
        n_classes=2,
        skip_attention=args.skip_attention,
        partial_ratios=args.partial_ratios,
        gft_kind=args.gft_kind,
        gft_bottleneck=args.gft_bottleneck,
        upsample_kind=args.upsample_kind,
        se_kind=args.se_kind,
        filters=args.filters,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {args.resume}")

    criterion = UltimateCombinedLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        iou_weight=args.iou_weight,
        focal_weight=args.focal_weight,
        boundary_weight=args.boundary_weight,
        connectivity_weight=args.connectivity_weight,
        device=device,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = Path(args.log_dir) / f"lm_net_varroa_{timestamp}.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler, amp=use_amp)
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False, amp=use_amp)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Train - Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} "
            f"| IoU: {train_metrics['iou']:.4f} | Precision: {train_metrics['precision']:.4f} "
            f"| Recall: {train_metrics['recall']:.4f}"
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} "
            f"| IoU: {val_metrics['iou']:.4f} | Precision: {val_metrics['precision']:.4f} "
            f"| Recall: {val_metrics['recall']:.4f}"
        )
        print(
            f"Time  - Train: {train_seconds:.2f}s | Val: {val_seconds:.2f}s "
            f"| Epoch: {epoch_seconds:.2f}s"
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_metrics(csv_log_path, row)
        scheduler.step(val_metrics["loss"])
        early_stopping(val_metrics["loss"], model, epoch, val_metrics)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining completed.")
    print(f"CSV log saved at: {csv_log_path}")
    print(f"Best epoch: {early_stopping.best_epoch}")
    print(f"Best metrics: {early_stopping.best_metrics}")


if __name__ == "__main__":
    main()
