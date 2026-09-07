import torch

from ultralytics.nn.modules.block import AdversarialPerturbationInjection


def test_api_perturbation_norm_and_relative_diagnostic():
    api = AdversarialPerturbationInjection(4, rho=0.02, api_weight=0.25, target_mode="boxgrad")
    api.train()
    feature = torch.randn(2, 4, 8, 8, requires_grad=True)
    grad = torch.autograd.grad((feature.square()).mean(), feature)[0]

    assert api.set_perturbation_from_grad(grad)
    assert api.last_perturbation_norm is not None
    assert torch.allclose(api.last_perturbation_norm, torch.full((2,), 0.02), atol=1e-5)

    relative = api.last_perturbation_norm / feature.detach().flatten(1).norm(p=2, dim=1).clamp_min(api.eps)
    assert torch.isfinite(relative).all()
    assert (relative > 0).all()
