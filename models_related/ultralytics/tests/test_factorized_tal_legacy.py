import torch

from ultralytics.utils.loss import v8DetectionLoss


def test_legacy_factorized_tal_matches_historical_formula_and_ignores_iou_ceiling():
    loss = object.__new__(v8DetectionLoss)
    loss.factorized_tal_mode = "legacy"
    loss.factorized_tal_tau = 0.75
    loss.factorized_tal_kappa = 1.5

    q = torch.tensor([[0.65, 0.40, 0.20], [0.35, 0.10, 0.0]])
    u = torch.tensor([0.99, 0.98])
    lam = 0.5
    q_max = q.max().clamp_min(1e-12)
    historical = q_max.pow(0.75) * (q / q_max).clamp(0, 1).pow(1.5)
    expected = q + lam * (torch.where(q > 0, historical, q) - q)

    actual, metrics = loss.factorize_tal_targets(q, u, lam)
    torch.testing.assert_close(actual, expected)
    assert metrics == {}


def test_legacy_factorized_tal_is_distinct_from_current_iou_ceiling():
    loss = object.__new__(v8DetectionLoss)
    loss.factorized_tal_tau = 0.75
    loss.factorized_tal_kappa = 1.5
    q = torch.tensor([[0.65], [0.40]])
    u = torch.tensor([0.99, 0.98])

    loss.factorized_tal_mode = "legacy"
    legacy, _ = loss.factorize_tal_targets(q, u, 0.5)
    loss.factorized_tal_mode = "current"
    current, _ = loss.factorize_tal_targets(q, u, 0.5)

    assert not torch.equal(legacy, current)
