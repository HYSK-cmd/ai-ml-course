from __future__ import annotations

import numpy as np

from forge.autograd import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy. logits (N, C), targets int (N,). Must be stable for huge logits."""
    raise NotImplementedError


def mse_loss(pred: Tensor, target) -> Tensor:
    """Mean squared error over all elements."""
    raise NotImplementedError
