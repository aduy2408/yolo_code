"""Small evidence branches used by the LEVIR P2 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EvidenceConv(nn.Sequential):
    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class GradientIsolatedEvidence(nn.Module):
    """Append a raw-image evidence branch, optionally isolating detection gradients."""

    def __init__(self, c1: int, evidence_ch: int = 8, detach_detection: bool = True, aux_enabled: bool = True) -> None:
        super().__init__()
        self.evidence_ch = int(evidence_ch)
        self.detach_detection = bool(detach_detection)
        self.aux_enabled = bool(aux_enabled)
        self.stem = nn.Sequential(_EvidenceConv(3, 16, 2), _EvidenceConv(16, self.evidence_ch, 2), _EvidenceConv(self.evidence_ch, self.evidence_ch))
        self.aux_head = nn.Conv2d(self.evidence_ch, 1, 1)
        self.last_aux: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, img0: torch.Tensor) -> torch.Tensor:
        evidence = self.stem(img0)
        if evidence.shape[-2:] != x.shape[-2:]:
            evidence = F.interpolate(evidence, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.training and self.aux_enabled:
            self.last_aux = {"evidence_heatmap": self.aux_head(evidence)}
        else:
            self.last_aux = None
        detection_evidence = evidence.detach() if self.detach_detection else evidence
        return torch.cat((x, detection_evidence), dim=1)


class AugmentationAwareEvidence(nn.Module):
    """Condition an image evidence branch on its predicted resolution state."""

    def __init__(self, c1: int, evidence_ch: int = 8) -> None:
        super().__init__()
        self.evidence_ch = int(evidence_ch)
        self.stem = nn.Sequential(_EvidenceConv(3, 16, 2), _EvidenceConv(16, self.evidence_ch, 2), _EvidenceConv(self.evidence_ch, self.evidence_ch))
        self.state_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.evidence_ch, 8), nn.SiLU(), nn.Linear(8, 1))
        self.film = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, self.evidence_ch * 2))
        self.last_aux: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, img0: torch.Tensor) -> torch.Tensor:
        base = self.stem(img0)
        if base.shape[-2:] != x.shape[-2:]:
            base = F.interpolate(base, size=x.shape[-2:], mode="bilinear", align_corners=False)
        state = torch.sigmoid(self.state_head(base))
        gamma, beta = self.film(state).chunk(2, dim=1)
        evidence = (1.0 + gamma[..., None, None]) * base + beta[..., None, None]
        self.last_aux = {"resolution_pred": state} if self.training else None
        return torch.cat((x, evidence), dim=1)


class ScaleDisappearanceEvidence(nn.Module):
    """Form an 8-channel evidence representation from detached backbone P2/P3."""

    def __init__(self, c_fine: int, c_coarse: int, out_ch: int = 8, hidden: int = 16, detach_sources: bool = True) -> None:
        super().__init__()
        self.detach_sources = bool(detach_sources)
        self.fine = nn.Sequential(nn.Conv2d(c_fine, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.coarse = nn.Sequential(nn.Conv2d(c_coarse, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.formation = nn.Sequential(_EvidenceConv(hidden * 4, hidden), _EvidenceConv(hidden, out_ch))
        self.last_aux = None

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fine, coarse = xs
        if self.detach_sources:
            fine, coarse = fine.detach(), coarse.detach()
        fine = self.fine(fine)
        coarse = self.coarse(coarse)
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        difference = fine - coarse
        return self.formation(torch.cat((fine, coarse, difference, difference.abs()), dim=1))
