#!/usr/bin/env python3
"""Resume only the corrected clean C2 and C3 complementary evidence runs."""
from misc import train_oaief_complement as base

base.EXPERIMENTS = {
    "C2": base.CFG / "yolov8n_p2_levir_complement_hard_negative.yaml",
    "C3": base.CFG / "yolov8n_p2_levir_complement_retrieval.yaml",
}

if __name__ == "__main__":
    base.run(base.args())
