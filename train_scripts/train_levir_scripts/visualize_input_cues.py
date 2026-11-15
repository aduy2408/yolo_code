#!/usr/bin/env python3
"""Save an RGB-plus-cue grid for visual normalization and boundary checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics.nn.modules import INPUT_CUE_VARIANTS, InputCueBank


def to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    return values.unsqueeze(0)


def render_channel(channel: torch.Tensor) -> Image.Image:
    channel = channel.detach().float().cpu()
    low, high = channel.quantile(0.01), channel.quantile(0.99)
    normalized = ((channel - low) / (high - low + 1e-6)).clamp(0, 1)
    return Image.fromarray((normalized.mul(255).byte().numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("input_cue_grid.png"))
    args = parser.parse_args()
    rgb = to_tensor(args.image)
    original = Image.open(args.image).convert("RGB")
    tiles = [("RGB", original)]
    for cue_type in INPUT_CUE_VARIANTS:
        cue = InputCueBank(cue_type)(rgb)[0]
        for index in range(cue.shape[0]):
            tiles.append((f"{cue_type}:{index}", render_channel(cue[index])))
    width = max(tile.width for _, tile in tiles)
    height = max(tile.height for _, tile in tiles)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + 24)), "white")
    for index, (label, tile) in enumerate(tiles):
        tile = ImageOps.fit(tile.convert("RGB"), (width, height))
        x = (index % columns) * width
        y = (index // columns) * (height + 24)
        canvas.paste(tile, (x, y + 24))
        ImageDraw.Draw(canvas).text((x + 4, y + 4), label, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
