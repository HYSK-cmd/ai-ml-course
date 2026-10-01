"""Layers. Parameters are float32 by default (like PyTorch); tests call ``.to(np.float64)`` for gradchecks.
Every layer's forward must be built from differentiable forge ops (or a custom op with a correct
backward) -- the tests gradcheck them."""
from __future__ import annotations

import numpy as np

from forge.autograd import Tensor
from forge.nn.module import Module


class Linear(Module):
    """y = x @ W.T + b. weight: Parameter (out_features, in_features), He init N(0, 2/in_features);
    bias: Parameter (out_features,) zeros, or None if bias=False."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, rng: np.random.Generator | None = None):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Flatten(Module):
    """(N, d1, d2, ...) -> (N, d1*d2*...)"""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Dropout(Module):
    """INVERTED dropout: in training zero each element with prob p and scale survivors by 1/(1-p);
    identity in eval mode."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class BatchNorm1d(Module):
    """Input (N, C). PyTorch semantics:
    train: normalize with batch mean and BIASED batch variance; update
           running_mean = (1-momentum)*running_mean + momentum*mean
           running_var  = (1-momentum)*running_var  + momentum*UNBIASED_var
    eval:  normalize with running stats.
    Parameters: weight (gamma, ones), bias (beta, zeros). Buffers: running_mean (zeros), running_var (ones)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LayerNorm(Module):
    """Normalize over the LAST dimension (biased variance), then scale/shift by weight/bias (size = normalized_shape).
    Same in train and eval."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
