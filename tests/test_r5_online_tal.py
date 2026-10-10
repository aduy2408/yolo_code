import torch

from project_ultralytics.r5 import (
    R5EpochFeedbackAdapter,
    ScaleBinSpec,
    ScaleDifficultyController,
    TALDifficultyCollector,
)



def _inputs():
    pred_bboxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 14.0, 14.0], [20.0, 20.0, 24.0, 24.0]]]
    )
    pred_scores = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 14.0, 14.0], [30.0, 30.0, 34.0, 34.0]]])
    gt_labels = torch.tensor([[[0], [1], [0]]])
    fg_mask = torch.tensor([[True, True, False]])
    target_gt_idx = torch.tensor([[0, 1, 2]])
    valid_gt_mask = torch.tensor([[[True], [True], [True]]])
    return pred_bboxes, pred_scores, gt_bboxes, gt_labels, fg_mask, target_gt_idx, valid_gt_mask


def test_tal_iou_collector_reports_missing_positive_as_max_difficulty():
    collector = TALDifficultyCollector(ScaleBinSpec((0, 8, 12)), mode="iou")
    batch = collector.collect(*_inputs())
    assert torch.allclose(batch.quality, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(batch.difficulty, torch.tensor([[0.0, 0.0, 1.0]]))
    assert batch.values.tolist() == [0.0, 0.0, 1.0]
    assert batch.bin_ids.tolist() == [0, 0, 0]


def test_tal_alignment_collector_multiplies_class_score_and_iou():
    collector = TALDifficultyCollector(ScaleBinSpec((0, 8, 12)), mode="alignment")
    batch = collector.collect(*_inputs())
    assert torch.allclose(batch.quality, torch.tensor([[0.9, 0.2, 0.0]]))
    assert torch.allclose(batch.difficulty, torch.tensor([[0.1, 0.8, 1.0]]))


def test_tal_collector_excludes_synthetic_gt_from_observations():
    collector = TALDifficultyCollector(ScaleBinSpec((0, 8, 12)), mode="iou")
    inputs = list(_inputs())
    original_gt_mask = torch.tensor([[True, False, True]])
    batch = collector.collect(*inputs, original_gt_mask=original_gt_mask)
    assert batch.valid_gt.tolist() == [[True, False, True]]
    assert batch.values.tolist() == [0.0, 1.0]


def test_epoch_adapter_accumulates_and_updates_sampler():
    class DummyTransform:
        def __init__(self):
            self.received = None

        def end_epoch_from_controller(self, controller, frequency=None, hybrid_ratio=1.0):
            self.received = controller.probabilities(feasible=[True, True, True, True])
            controller.reset_pending()
            return self.received.tolist()

    controller = ScaleDifficultyController(num_bins=4, beta=0.0, exploration=0.0)
    adapter = R5EpochFeedbackAdapter(
        DummyTransform(), controller,
        collector=TALDifficultyCollector(ScaleBinSpec((0, 8, 12, 16, 20)), mode="iou"),
    )
    adapter.observe(
        pred_bboxes=_inputs()[0],
        pred_scores=_inputs()[1],
        gt_bboxes=_inputs()[2],
        gt_labels=_inputs()[3],
        fg_mask=_inputs()[4],
        target_gt_idx=_inputs()[5],
        valid_gt_mask=_inputs()[6],
    )
    probabilities = adapter.end_epoch()
    assert adapter.observations == 3
    assert len(probabilities) == 4
    assert torch.isfinite(torch.tensor(probabilities)).all()
