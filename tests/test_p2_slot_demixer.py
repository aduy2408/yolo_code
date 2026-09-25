from pathlib import Path

import torch

from project_ultralytics import load_project_model
from project_ultralytics.assignment import CollisionPreservingTaskAlignedAssigner


ROOT = Path(__file__).resolve().parents[1]


def test_p2_slot_head_has_distinct_final_predictors() -> None:
    model = load_project_model(
        ROOT / "project_ultralytics/configs/p2_slots/yolov8n_tinyperson_p2p3p4_b1_capacity.yaml",
        task="detect",
        verbose=False,
    )
    head = model.model.model[-1]
    assert head.cv2[0].predictors[0] is not head.cv2[0].predictors[1]
    assert head.cv3[0].predictors[0] is not head.cv3[0].predictors[1]


def test_collision_assigner_keeps_two_slots_at_one_spatial_location() -> None:
    assigner = CollisionPreservingTaskAlignedAssigner(
        topk=1, num_classes=1, alpha=0.5, beta=1.0, stride=[4, 8, 16]
    )
    assigner.p2_base_count = 2
    assigner.p2_slot_count = 2
    scores = torch.full((1, 4, 1), 0.9)
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 4])
    anchors = torch.tensor([[1.0, 1.0], [3.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    labels = torch.zeros((1, 2, 1), dtype=torch.long)
    gt_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]])
    mask_gt = torch.ones((1, 2, 1), dtype=torch.bool)

    _, _, _, fg_mask, target_gt_idx = assigner(scores, boxes, anchors, labels, gt_boxes, mask_gt)

    assert fg_mask[0, 0] and fg_mask[0, 2]
    assert not fg_mask[0, 1] and not fg_mask[0, 3]
    assert target_gt_idx[0, 0] != target_gt_idx[0, 2]
