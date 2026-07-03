from pathlib import Path
import json
import os
import subprocess

paths = [
    Path("/marimo/runs/detect/rank_loss_staged/baseline40_then_rank005.status.json"),
    Path("/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep/weights/best.pt"),
    Path("/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep/weights/last.pt"),
]
for p in paths:
    print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else None)
    if p.exists() and p.suffix == ".json":
        print(p.read_text()[:2000])
print("rank_loss_staged dirs")
for p in sorted(Path("/marimo/runs/detect/rank_loss_staged").glob("*")):
    print(p, "dir" if p.is_dir() else "file")
print("rank_loss_real dirs")
for p in sorted(Path("/marimo/runs/detect/rank_loss_real").glob("*")):
    print(p, "dir" if p.is_dir() else "file")
