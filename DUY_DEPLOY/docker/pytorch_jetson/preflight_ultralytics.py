from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO

root = Path('/checkpoints')
paths = sorted(root.glob('*/*/seed_42/best.pt'))
if len(paths) != 15:
    raise RuntimeError(f'expected 15 checkpoints, found {len(paths)}')
rows = []
for path in paths:
    model = YOLO(str(path))
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    rows.append({'checkpoint': str(path), 'parameters': parameters})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0), 'loaded': len(rows)}), flush=True)
