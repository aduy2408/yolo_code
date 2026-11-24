"""MuSGD optimizer adapter for the project-owned MMDetection runner.

This follows the pinned Ultralytics MuSGD recipe: lr=0.01, momentum=0.9,
nesterov=True, Muon weight=0.2, and SGD weight=1.0. Matrix and convolution
weights use the Muon branch. Biases and normalization parameters use SGD.
"""

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(gradient: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    matrix = gradient.reshape(-1, gradient.size(-2), gradient.size(-1)).bfloat16()
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = gradient.size(-2) > gradient.size(-1)
    if transposed:
        matrix = matrix.transpose(-2, -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        gram = matrix @ matrix.transpose(-2, -1)
        correction = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        matrix = torch.baddbmm(matrix, correction, matrix, beta=a)
    if transposed:
        matrix = matrix.transpose(-2, -1)
    return matrix.reshape(gradient.shape)


def muon_update(gradient: torch.Tensor, momentum: torch.Tensor, beta: float, nesterov: bool) -> torch.Tensor:
    gradient = gradient.detach().contiguous()
    momentum.mul_(beta).add_(gradient, alpha=1 - beta)
    update = gradient.lerp(momentum, beta) if nesterov else momentum
    reshaped = update.view(len(update), -1) if update.ndim > 2 else update
    result = zeropower_via_newtonschulz5(reshaped)
    result = result * max(1, gradient.size(-2) / gradient.size(-1)) ** 0.5
    return result.reshape(gradient.shape).to(gradient.dtype)


class MuSGD(optim.Optimizer):
    """Hybrid Muon plus SGD optimizer compatible with MMEngine."""

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        nesterov: bool = True,
        use_muon: bool = False,
        muon: float = 0.2,
        sgd: float = 1.0,
        **kwargs,
    ):
        del kwargs
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                use_muon=use_muon,
            ),
        )
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().contiguous()
                state = self.state[parameter]
                if group["use_muon"] and parameter.ndim >= 2:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                        state["momentum_buffer_sgd"] = torch.zeros_like(parameter)
                    update = muon_update(
                        gradient,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    parameter.add_(update, alpha=-(group["lr"] * self.muon))
                    sgd_gradient = gradient.add(parameter, alpha=group["weight_decay"])
                    state["momentum_buffer_sgd"].mul_(group["momentum"]).add_(sgd_gradient)
                    sgd_update = (
                        sgd_gradient.add(state["momentum_buffer_sgd"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_sgd"]
                    )
                    parameter.add_(sgd_update, alpha=-(group["lr"] * self.sgd))
                else:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    sgd_gradient = gradient.add(parameter, alpha=group["weight_decay"])
                    state["momentum_buffer"].mul_(group["momentum"]).add_(sgd_gradient)
                    update = (
                        sgd_gradient.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    parameter.add_(update, alpha=-group["lr"])
        return loss


def register_musgd() -> None:
    """Register MuSGD in the MMDetection optimizer registry once."""
    from mmdet.registry import OPTIMIZERS

    if OPTIMIZERS.get("MuSGD") is not None:
        return

    @OPTIMIZERS.register_module(name="MuSGD")
    class RegisteredMuSGD(MuSGD):
        def __init__(self, params, **kwargs):
            incoming = list(params)
            expanded = []
            for item in incoming:
                group = dict(item) if isinstance(item, dict) else {"params": [item]}
                parameters = list(group.pop("params"))
                explicit = group.pop("use_muon", None)
                buckets = (
                    [(bool(explicit), parameters)]
                    if explicit is not None
                    else [
                        (True, [p for p in parameters if p.ndim >= 2]),
                        (False, [p for p in parameters if p.ndim < 2]),
                    ]
                )
                for use_muon, selected in buckets:
                    if selected:
                        part = dict(group)
                        part["params"] = selected
                        part["use_muon"] = use_muon
                        expanded.append(part)
            super().__init__(expanded, **kwargs)
