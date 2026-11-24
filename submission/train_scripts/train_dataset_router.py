#!/usr/bin/env python3
"""Train a lightweight Varroa/LEVIR-Ship/TinyPerson image router.

This first experiment routes full images, not detector crops. It uses deterministic
source-aware split hints where available and saves a loadable checkpoint plus JSON
metadata. For production, replace the LEVIR hash split with scene-disjoint groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

CLASSES = ("varroa", "levirship", "tinyperson")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def bucket(path: Path) -> float:
    return int(hashlib.sha1(str(path).encode()).hexdigest()[:8], 16) / 2**32


def collect_records(data_root: Path, seed: int) -> dict[str, dict[str, list[Path]]]:
    records = {c: {s: [] for s in ("train", "val", "test")} for c in CLASSES}
    varroa = data_root / "Varroa"
    for split in records["varroa"]:
        records["varroa"][split] = image_files(varroa / split)
    for path in image_files(data_root / "LevirShip"):
        split = "train" if bucket(path) < 0.70 else ("val" if bucket(path) < 0.85 else "test")
        records["levirship"][split].append(path)
    for path in image_files(data_root / "TinyPerson"):
        if "test" in path.parts:
            records["tinyperson"]["test"].append(path)
        elif "train" in path.parts:
            split = "val" if bucket(path) >= 0.85 else "train"
            records["tinyperson"][split].append(path)
    for class_name in CLASSES:
        for split in records[class_name]:
            random.Random(seed + sum(map(ord, class_name + split))).shuffle(records[class_name][split])
    return records


class ImageSet(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform: T.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), label


class TinyRouter(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 160, 3, 2, 1), nn.BatchNorm2d(160), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(160, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    labels, predictions, confidences = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            probabilities = model(images.to(device)).softmax(1)
            labels.extend(targets.tolist())
            predictions.extend(probabilities.argmax(1).cpu().tolist())
            confidences.extend(probabilities.max(1).values.cpu().tolist())
    confusion = [[sum(y == i and p == j for y, p in zip(labels, predictions)) for j in range(3)] for i in range(3)]
    recall = [confusion[i][i] / max(1, sum(confusion[i])) for i in range(3)]
    return {
        "accuracy": sum(y == p for y, p in zip(labels, predictions)) / max(1, len(labels)),
        "per_class_recall": dict(zip(CLASSES, recall)),
        "confusion_matrix": confusion,
        "mean_confidence": sum(confidences) / max(1, len(confidences)),
        "n": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    records = collect_records(args.data_root, args.seed)
    train_n = min(args.max_per_class, min(len(records[c]["train"]) for c in CLASSES))
    for class_name in CLASSES:
        records[class_name]["train"] = records[class_name]["train"][:train_n]
        records[class_name]["val"] = records[class_name]["val"][:200]
        records[class_name]["test"] = records[class_name]["test"][:200]
        print(class_name, {split: len(records[class_name][split]) for split in records[class_name]})
    train_transform = T.Compose([T.RandomResizedCrop(args.input_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(), T.ColorJitter(0.25, 0.25, 0.25, 0.08), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_transform = T.Compose([T.Resize(round(args.input_size * 1.125)), T.CenterCrop(args.input_size), T.ToTensor(), T.Normalize(MEAN, STD)])
    def make(split, transform):
        return ImageSet([(p, i) for i, c in enumerate(CLASSES) for p in records[c][split]], transform)
    train_loader = DataLoader(make("train", train_transform), 128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(make("val", eval_transform), 256, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(make("test", eval_transform), 256, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            total_loss += loss.detach().item() * len(labels); count += len(labels)
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / count, "val": validation}
        history.append(row); print(row)
    test = evaluate(model, test_loader, device)
    metadata = {"classes": list(CLASSES), "architecture": "TinyRouter-v1", "input_size": args.input_size, "mean": list(MEAN), "std": list(STD), "seed": args.seed, "train_per_class": train_n, "history": history, "test": test, "nms_iou": 0.5, "split_note": "LEVIR uses deterministic hash split; replace with scene-disjoint split for production."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"artifact": str(args.output), "parameters": sum(p.numel() for p in model.parameters()), "test": test}, indent=2))


if __name__ == "__main__":
    main()
