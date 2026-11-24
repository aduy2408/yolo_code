#!/usr/bin/env python3
"""Build a draw.io resource pack from exact LevirShip copy-paste provenance."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "docs/reports/augmentation_examples/levir_cp_donor_provenance.json"
OUT = ROOT / "docs/reports/augmentation_examples/drawio_levirship_resource_pack"


def read_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_box(image, box, pad=8):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(width, x2 + pad)
    bottom = min(height, y2 + pad)
    return image[top:bottom, left:right], (left, top)


def save_png(image, path: Path, title: str | None = None, box=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    ax.imshow(image)
    if box is not None:
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#e31b23", linewidth=3))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, weight="bold")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def page(pdf, panels, title):
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3), dpi=160)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=16, weight="bold")
    for ax, panel in zip(axes, panels):
        image, label, box = panel
        ax.imshow(image)
        if box is not None:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#e31b23", linewidth=2))
        ax.set_title(label, fontsize=9, weight="bold")
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main():
    metadata = json.loads(META.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    target = read_rgb(Path(metadata["target_image"]))
    panels = [(target, "Target canvas / image without objects", None)]
    crops = []
    for item in metadata["variants"]:
        donor_path = Path(item["donor_image"])
        donor = read_rgb(donor_path)
        bbox = item["donor_bbox_xyxy"]
        crop, crop_origin = crop_box(donor, bbox, pad=12)
        local_box = [
            bbox[0] - crop_origin[0],
            bbox[1] - crop_origin[1],
            bbox[2] - crop_origin[0],
            bbox[3] - crop_origin[1],
        ]
        name = item["name"].lower().replace(" ", "_")
        save_png(donor, OUT / f"{name}_donor_image.png", item["name"] + " donor image", box=bbox)
        save_png(crop, OUT / f"{name}_object_crop.png", item["name"] + " object crop", box=local_box)
        panels.append((donor, item["name"] + " donor image", bbox))
        crops.append((crop, item["name"] + " extracted object crop", local_box))

    save_png(target, OUT / "target_negative_canvas.png", "Target negative canvas")

    with PdfPages(OUT / "levirship_copy_paste_drawio_resources.pdf") as pdf:
        page(pdf, panels[:6], "LevirShip source images and selected donor objects")
        page(pdf, crops + [(target, "Negative target canvas", None)], "LevirShip extracted object crops and target canvas")

        existing = ROOT / "docs/reports/augmentation_examples/levir_cp_two_sections_combined.png"
        if existing.exists():
            figure = read_rgb(existing)
            fig, ax = plt.subplots(figsize=(11.7, 16.5), dpi=140)
            ax.imshow(figure)
            ax.axis("off")
            ax.set_title("Existing exact-provenance copy-paste flow", fontsize=16, weight="bold")
            fig.tight_layout()
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

    manifest = {
        "source_dataset": "LevirShipData",
        "provenance": str(META.relative_to(ROOT)),
        "target_image": metadata["target_image"],
        "resources": sorted(path.name for path in OUT.glob("*.png")),
        "pdf": "levirship_copy_paste_drawio_resources.pdf",
        "note": "All donor images and crops are taken from the exact ObjectRecord provenance used by the project figures.",
    }
    (OUT / "README.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT / 'levirship_copy_paste_drawio_resources.pdf'}")


if __name__ == "__main__":
    main()
