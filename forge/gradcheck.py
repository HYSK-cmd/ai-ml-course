"""Week 1 -- numerical gradient checking. Your best debugging tool for the rest of the course."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from forge.autograd import Tensor


def numerical_grad(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], eps: float = 1e-6) -> list[np.ndarray]:
    """Central differences: d f / d inputs[i] for every element of every input.

    ``f`` takes Tensors (one per input array, same order) and returns a scalar Tensor.
    ``inputs`` are float64 arrays; do not leave them modified when you return.
    """
    raise NotImplementedError


def gradcheck(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], eps: float = 1e-6,
              atol: float = 1e-5, rtol: float = 1e-3) -> bool:
    """True iff autograd gradients of ``f`` match ``numerical_grad`` (np.allclose with atol/rtol)
    for every input. Wrap each input as ``Tensor(x.copy(), requires_grad=True)`` for the analytic pass."""
    raise NotImplementedError
