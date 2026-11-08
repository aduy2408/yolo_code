"""Inference-only compatibility defaults for older serialized checkpoints."""

from __future__ import annotations


def patch_legacy_modules(model) -> list[str]:
    patched: list[str] = []
    inner = getattr(model, "model", model)
    modules = inner.modules() if hasattr(inner, "modules") else ()
    for module in modules:
        if type(module).__name__ == "SRMF2Guidance" and not hasattr(module, "use_saturation_guidance"):
            module.use_saturation_guidance = bool(
                hasattr(module, "saturation_mask_net") and hasattr(module, "gamma_saturation_raw")
            )
            patched.append(type(module).__name__)
    return patched
