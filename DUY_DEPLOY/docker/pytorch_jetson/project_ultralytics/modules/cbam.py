"""Project-specific CBAM attention primitives."""

from __future__ import annotations

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels from global average, max, or both descriptors.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int, descriptor: str = "avg", detach_descriptor: bool = False) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
            descriptor (str): Global descriptor: ``avg``, ``max``, or ``avg_max``.
            detach_descriptor (bool): Detach descriptor input while preserving ``x * gate`` forward values.
        """
        super().__init__()
        if descriptor not in {"avg", "max", "avg_max"}:
            raise ValueError(f"Unsupported channel descriptor: {descriptor!r}")
        self.descriptor = descriptor
        self.detach_descriptor = detach_descriptor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1) if descriptor != "avg" else None
        self.max_fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True) if descriptor == "avg_max" else None
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        descriptor_x = x.detach() if getattr(self, "detach_descriptor", False) and self.training else x
        average = self.fc(self.pool(descriptor_x))
        descriptor = getattr(self, "descriptor", "avg")
        if descriptor == "avg":
            gate = average
        else:
            maximum = self.fc(self.max_pool(descriptor_x)) if descriptor == "max" else self.max_fc(self.max_pool(descriptor_x))
            gate = maximum if descriptor == "max" else average + maximum
        act_gate = self.act(gate)
        if getattr(self, "override_gate_fn", None) is not None:
            act_gate = self.override_gate_fn(self, act_gate)
        return x * act_gate

class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))
