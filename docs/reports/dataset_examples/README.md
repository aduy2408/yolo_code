# Dataset examples

These figures show ground-truth YOLO bounding boxes in red for three validation images from each dataset:

- `varroa/`: Varroa, 1/2/3 annotations
- `levir_ship/`: LEVIR-Ship, 1/2/4 ship annotations
- `tiny_person/`: TinyPerson, 1/3/8 person annotations

Each directory contains `example_1.png` through `example_3.png` and a `contact_sheet.png` for direct use in the report.

Recreate them from the repository root with:

```bash
source /home/duylearch/miniconda3/bin/activate ml2
python misc/plot_dataset_examples.py
```

The default source is the validation split. Use `--split train` or `--split test` to select another split.
