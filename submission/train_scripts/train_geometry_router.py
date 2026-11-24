#!/usr/bin/env python3
"""Train a tiny dataset router from native image aspect ratio.

This is intentionally a geometry router: deployment must pass the original image
geometry to it before dataset-specific detector preprocessing.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

CLASSES = ("varroa", "levirship", "tinyperson")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)

def bucket(path):
    return int(hashlib.sha1(str(path).encode()).hexdigest()[:8], 16) / 2**32

def records(root, seed):
    r = {c: {s: [] for s in ("train", "test")} for c in CLASSES}
    for split in r["varroa"]: r["varroa"][split] = files(root / "Varroa" / split)
    for p in files(root / "LevirShip"):
        r["levirship"]["train" if bucket(p) < .85 else "test"].append(p)
    for p in files(root / "TinyPerson"):
        if "test" in p.parts: r["tinyperson"]["test"].append(p)
        elif "train" in p.parts: r["tinyperson"]["train"].append(p)
    for c in CLASSES:
        for s in r[c]: random.Random(seed + sum(map(ord, c + s))).shuffle(r[c][s])
    return r

def features(paths):
    return [[Image.open(p).width / Image.open(p).height] for p in paths]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
    r = records(args.data_root, args.seed)
    train_paths = [p for c in CLASSES for p in r[c]["train"]]; train_y = [i for i,c in enumerate(CLASSES) for _ in r[c]["train"]]
    test_paths = [p for c in CLASSES for p in r[c]["test"]]; test_y = [i for i,c in enumerate(CLASSES) for _ in r[c]["test"]]
    model = DecisionTreeClassifier(max_depth=4, random_state=args.seed).fit(features(train_paths), train_y)
    pred = model.predict(features(test_paths))
    metadata = {"classes": list(CLASSES), "feature": "native_width_divided_by_height", "max_depth": 4, "seed": args.seed, "tree": export_text(model, feature_names=["aspect_ratio"]), "test_counts": {c: len(r[c]["test"]) for c in CLASSES}, "test_accuracy": float(accuracy_score(test_y, pred)), "confusion_matrix": confusion_matrix(test_y, pred).tolist(), "parameters": int(model.tree_.node_count), "nms_iou": 0.5}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
