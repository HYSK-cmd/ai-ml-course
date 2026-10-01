"""Week 1 -- optimizers. Semantics match PyTorch exactly (the tests use numbers produced by PyTorch's
formulas), so derive each update rule on paper first.

All optimizers:
* take an iterable of Tensors (materialize it -- generators are allowed),
* skip parameters whose ``.grad is None``,
* update ``p.data`` IN PLACE (other code holds references to the same Tensor objects),
* ``zero_grad()`` sets every ``p.grad = None``.
"""
from __future__ import annotations

from collections.abc import Iterable

from forge.autograd import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """g = grad + weight_decay * p;  v = momentum * v + g  (v starts as g on the first step);
    p -= lr * v.   (momentum=0 means plain SGD.)"""

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class Adam(Optimizer):
    """Adam with bias correction (Kingma & Ba). No weight decay."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class AdamW(Adam):
    """Adam + DECOUPLED weight decay: before the Adam update, p -= lr * weight_decay * p."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError
