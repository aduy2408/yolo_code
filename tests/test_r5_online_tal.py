import torch
from torch.utils.data import DataLoader, Dataset

from project_ultralytics.r5 import (
    R5EpochFeedbackAdapter,
    ScaleBinSampler,
    ScaleBinSpec,
    ScaleDifficultyController,
    TALDifficultyCollector,
    register_r5_epoch_callback,
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


def test_alignment_collector_can_sigmoid_logits_once():
    collector = TALDifficultyCollector(ScaleBinSpec((0, 8, 12)), mode="alignment", scores_are_logits=True)
    inputs = list(_inputs())
    inputs[1] = torch.logit(inputs[1])
    batch = collector.collect(*inputs)
    assert torch.allclose(batch.quality, torch.tensor([[0.9, 0.2, 0.0]]), atol=1e-6)


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


def test_epoch_callback_registration_is_idempotent_and_updates_once():
    class DummyTransform:
        def __init__(self):
            self.calls = 0

        def end_epoch_from_controller(self, controller, frequency=None, hybrid_ratio=1.0):
            self.calls += 1
            controller.end_epoch()
            return [0.25, 0.25, 0.25, 0.25]

    class CallbackOwner:
        def __init__(self):
            self.callbacks = {"on_train_epoch_end": []}

        def add_callback(self, event, callback):
            self.callbacks[event].append(callback)

    owner = CallbackOwner()
    adapter = R5EpochFeedbackAdapter(DummyTransform(), ScaleDifficultyController(num_bins=4))
    assert register_r5_epoch_callback(owner, adapter)
    assert register_r5_epoch_callback(owner, adapter)
    assert len(owner.callbacks["on_train_epoch_end"]) == 1
    owner.callbacks["on_train_epoch_end"][0](owner)
    assert adapter.transform.calls == 1
    assert adapter.controller.epoch == 1


def test_project_detection_trainer_registers_r5_callback(monkeypatch):
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from project_ultralytics.training import ProjectDetectionTrainer

    def fake_init(self, *args, **kwargs):
        self.callbacks = {"on_train_epoch_end": []}

    monkeypatch.setattr(DetectionTrainer, "__init__", fake_init)
    adapter = R5EpochFeedbackAdapter(
        transform=type("Transform", (), {"end_epoch_from_controller": lambda *args, **kwargs: []})(),
        controller=ScaleDifficultyController(num_bins=4),
    )
    trainer = ProjectDetectionTrainer(r5_feedback_adapter=adapter)
    assert trainer.callbacks["on_train_epoch_end"] == [adapter.on_train_epoch_end]


def test_feedback_callback_updates_shared_state_seen_by_real_workers():
    class WorkerSamplingDataset(Dataset):
        def __init__(self, sampler):
            self.sampler = sampler
            self.values = {0: [4.0], 1: [], 2: [], 3: [16.0]}

        def __len__(self):
            return 64

        def __getitem__(self, index):
            return self.sampler.sample(self.values)

    class TransformBridge:
        def __init__(self, sampler):
            self.sampler = sampler

        def end_epoch_from_controller(self, controller, frequency=None, hybrid_ratio=1.0):
            controller.end_epoch()
            probabilities = controller.probabilities(feasible=[True, False, False, True])
            self.sampler.set_probabilities(probabilities.tolist())
            return probabilities.tolist()

    sampler = ScaleBinSampler(
        ScaleBinSpec((0, 8, 12, 16, 20)),
        seed=123,
        probabilities=[1.0, 0.0, 0.0, 0.0],
        enable_shared_state=True,
    )
    adapter = R5EpochFeedbackAdapter(
        TransformBridge(sampler),
        ScaleDifficultyController(num_bins=4, beta=0.0, exploration=0.0),
        collector=TALDifficultyCollector(ScaleBinSpec((0, 8, 12, 16, 20)), mode="iou"),
    )
    dataset = WorkerSamplingDataset(sampler)
    first = next(iter(DataLoader(dataset, batch_size=64, num_workers=2)))
    assert torch.all(first == 4.0)
    inputs = _inputs()
    adapter.observe(
        pred_bboxes=inputs[0],
        pred_scores=inputs[1],
        gt_bboxes=inputs[2],
        gt_labels=inputs[3],
        fg_mask=inputs[4],
        target_gt_idx=inputs[5],
        valid_gt_mask=inputs[6],
    )
    probabilities = adapter.on_train_epoch_end()
    second = next(iter(DataLoader(dataset, batch_size=64, num_workers=2)))
    assert adapter.observations == 3
    assert adapter.controller.epoch == 1
    assert probabilities[3] > 0.0
    assert torch.any(second == 16.0)
    assert not torch.all(second == 4.0)


def test_shared_probability_state_reaches_real_dataloader_workers():
    class WorkerSamplingDataset(Dataset):
        def __init__(self, sampler):
            self.sampler = sampler
            self.values = {0: [4.0], 1: [], 2: [], 3: [16.0]}

        def __len__(self):
            return 64

        def __getitem__(self, index):
            return self.sampler.sample(self.values)

    sampler = ScaleBinSampler(
        ScaleBinSpec((0, 8, 12, 16, 20)),
        seed=123,
        probabilities=[1.0, 0.0, 0.0, 0.0],
        enable_shared_state=True,
    )
    first = next(iter(DataLoader(WorkerSamplingDataset(sampler), batch_size=64, num_workers=2)))
    assert torch.all(first == 4.0)
    sampler.set_probabilities([0.0, 0.0, 0.0, 1.0])
    second = next(iter(DataLoader(WorkerSamplingDataset(sampler), batch_size=64, num_workers=2)))
    assert torch.all(second == 16.0)
