"""Convolution and pooling on (N, C, H, W) tensors.

Two valid designs -- pick one and be able to defend it:
  (a) compose existing autograd ops (pad + fancy-index im2col + reshape + matmul). Backward is free,
      but fancy-index backward (np.add.at) is slow.
  (b) a custom autograd op: im2col in forward (np.lib.stride_tricks.sliding_window_view), and a
      hand-written col2im in backward. Faster, and Week 5 asks you to make it >= 5x faster anyway.
Output size: H_out = (H + 2*padding - kernel_size) // stride + 1.
"""
from __future__ import annotations

import numpy as np

from forge.autograd import Tensor
from forge.nn.module import Module


class Conv2d(Module):
    """Cross-correlation (like every DL framework). weight: (out_channels, in_channels, k, k) with He
    init N(0, 2/(in_channels*k*k)); bias: (out_channels,) zeros."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, rng: np.random.Generator | None = None):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MaxPool2d(Module):
    """stride defaults to kernel_size. No padding. Gradient flows only to the max element of each
    window (if a window has ties, send it to exactly one of them)."""

    def __init__(self, kernel_size: int, stride: int | None = None):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
