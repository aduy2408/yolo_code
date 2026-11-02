from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx
from onnx.utils import extract_model

MODEL_HEAD = re.compile(r"/model\.(\d+)/((?:one2one_)?cv[23])\.(\d+)/")

def image_size(path: Path) -> int:
    return 512 if "levirship" in path.parts else 640


def prepare(source: Path, destination: Path) -> int:
    models = sorted(source.rglob("*.onnx"))
    if not models:
        raise FileNotFoundError(f"No ONNX models under {source}")
    for path in models:
        size = image_size(path)
        model = onnx.load(str(path), load_external_data=False)
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        input_shape[0].dim_value = 0
        input_shape[0].dim_param = "batch"
        input_shape[2].dim_param = ""
        input_shape[2].dim_value = size
        input_shape[3].dim_param = ""
        input_shape[3].dim_value = size
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        static_path = target.with_suffix(".static.onnx")
        onnx.save(model, str(static_path))
        heads: dict[tuple[str, int], tuple[int, str]] = {}
        for node in model.graph.node:
            match = MODEL_HEAD.search(node.name)
            if match and node.op_type == "Conv" and node.name.endswith("/Conv"):
                key = (match.group(2), int(match.group(3)))
                candidate = (int(match.group(1)), node.output[0])
                if key not in heads or candidate[0] >= heads[key][0]:
                    heads[key] = candidate
        branch_pair = "one2one_" if any(key[0].startswith("one2one_") for key in heads) else ""
        outputs = [
            heads[(f"{branch_pair}cv{branch}", scale)][1]
            for scale in range(3)
            for branch in (2, 3)
            if (f"{branch_pair}cv{branch}", scale) in heads
        ]
        if len(outputs) == 6:
            extract_model(str(static_path), str(target), [item.name for item in model.graph.input], outputs)
            static_path.unlink()
        else:
            static_path.rename(target)
    return len(models)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(f"prepared={prepare(args.source, args.destination)}")


if __name__ == "__main__":
    main()
