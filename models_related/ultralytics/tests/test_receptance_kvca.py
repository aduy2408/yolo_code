from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import KVCompressedAttention, ReceptanceKVCompressedAttention, SurgicalPartialKVCompressedAttention
from ultralytics.nn.tasks import DetectionModel


CONFIG_DIR = Path(__file__).parents[2] / "models_config/yolov8/levir"


def test_receptance_kvca_identity_and_initial_gate():
    module = ReceptanceKVCompressedAttention(64, 64, num_heads=4, sr_ratio=4, mode="group_weight").eval()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = module(x)
        module.capture_receptance = True
        module(x)
    assert y.shape == x.shape
    assert torch.max((y - x).abs()).item() < 1e-6
    gate = module.last_receptance
    assert gate is not None and gate.shape == x.shape
    assert gate.mean().item() == pytest.approx(0.5, abs=1e-7)
    assert gate.std().item() == pytest.approx(0.0, abs=1e-7)


def test_receptance_kvca_gate_receives_gradient_after_output_branch_opens():
    module = ReceptanceKVCompressedAttention(8, 8, num_heads=2, sr_ratio=2, mode="group_weight").train()
    module.proj_bn.weight.data.fill_(1.0)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    module(x).square().mean().backward()
    assert module.receptance.weight.grad is not None
    assert torch.isfinite(module.receptance.weight.grad).all()


def test_receptance_kvca_yaml_parse_and_model_smoke():
    model = DetectionModel(CONFIG_DIR / "yolov8n_p2_surgical_a_p3_receptance_kvca.yaml", ch=3, nc=1, verbose=False).eval()
    attention = model.model[16]
    assert isinstance(attention, ReceptanceKVCompressedAttention)
    assert attention.c2 == 64 and attention.num_heads == 4 and attention.sr_ratio == 4
    assert model.model[-1].f == [20]
    assert model.model[-1].stride.tolist() == [4.0]
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None


def test_receptance_kvca_preserves_common_attention_core():
    torch.manual_seed(7)
    bare = KVCompressedAttention(8, 8, num_heads=2, sr_ratio=2, mode="group_weight").eval()
    gated = ReceptanceKVCompressedAttention(8, 8, num_heads=2, sr_ratio=2, mode="group_weight").eval()
    common = {key: value for key, value in bare.state_dict().items() if key in gated.state_dict()}
    gated.load_state_dict(common, strict=False)
    with torch.no_grad():
        gated.receptance.weight.zero_()
        gated.receptance.bias.fill_(20.0)
        bare.proj_bn.weight.fill_(1.0)
        bare.proj_bn.bias.zero_()
        gated.proj_bn.weight.copy_(bare.proj_bn.weight)
        gated.proj_bn.bias.copy_(bare.proj_bn.bias)
    x = torch.randn(1, 8, 8, 8)
    assert torch.allclose(gated(x), bare(x), atol=1e-5, rtol=1e-5)


def test_surgical_partial_kvca_is_identity_safe_and_keeps_local_bypass():
    module = SurgicalPartialKVCompressedAttention(64, 64, num_heads=4, sr_ratio=4, mode="group_weight").eval()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert torch.max((y - x).abs()).item() < 1e-6
    assert module.attn.num_heads == 2
    assert not hasattr(module, "out_proj")
