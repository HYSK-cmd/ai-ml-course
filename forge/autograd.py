"""Week 1 -- reverse-mode automatic differentiation over numpy arrays.

Everything later in the course (forge.ml, forge.nn) is built on this file, so make it solid.

Contract (the tests rely on exactly this):

* ``Tensor(data, requires_grad=False)`` wraps ``np.asarray(data)``. If ``data`` is already a
  floating-point ndarray keep its dtype, otherwise convert to ``float64``.
* ``t.data`` is the ndarray, ``t.grad`` is ``None`` until backward writes an ndarray of the same shape.
* Every op returns a new Tensor. ``out.requires_grad`` is True iff any input requires grad
  (and grad mode is enabled -- see ``no_grad``).
* Python scalars and ndarrays mixed into ops are treated as constants (wrap them in a Tensor).
  Both orders must work: ``x * arr`` and ``arr * x``, ``x @ arr`` and ``arr @ x``. For the
  ndarray-on-the-left case numpy must defer to your reflected method -- that's what the
  ``__array_priority__`` class attribute below is for (leave it).
* ``backward()`` on a scalar (size-1) tensor seeds its grad with 1. On a non-scalar tensor you must
  pass ``grad=`` (an array of the same shape); otherwise raise ``RuntimeError``.
* Gradients ACCUMULATE into ``.grad`` of leaf tensors (tensors created by the user, not by an op)
  that have ``requires_grad=True``. Leaves with ``requires_grad=False`` keep ``grad is None``.
* Broadcasting: when an op broadcast an input, its gradient must be summed back to the input's
  shape -- implement ``unbroadcast`` and use it everywhere.
* ``backward()`` must NOT use recursion to build the topological order (graphs can be 10k+ deep).

Suggested internals (not tested directly): ``_prev`` (parents), ``_backward`` (closure that pushes
``out.grad`` into parents), ``_op`` (name, handy for debugging / drawing the graph).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing numpy broadcasting.

    Examples: grad (3, 4) -> shape (4,) sums axis 0; grad (3, 4) -> shape (3, 1) sums axis 1 with
    keepdims; grad (2, 3, 4) -> shape (1, 4) sums axes 0 and 1; shape () sums everything.
    """
    raise NotImplementedError


_GRAD_ENABLED = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Inside this block ops don't record a graph: outputs have requires_grad=False and no parents.
    Must restore the previous state even if the body raises, and must nest correctly."""
    raise NotImplementedError


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class Tensor:
    __array_priority__ = 100  # makes `ndarray <op> Tensor` call Tensor.__r<op>__

    def __init__(self, data, requires_grad: bool = False, _children: tuple[Tensor, ...] = (), _op: str = ""):
        raise NotImplementedError

    # ---- conveniences -------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def numpy(self) -> np.ndarray:
        """Return the underlying array (no copy)."""
        raise NotImplementedError

    def detach(self) -> Tensor:
        """New leaf tensor sharing data, requires_grad=False."""
        raise NotImplementedError

    # ---- elementwise arithmetic (all must broadcast) ------------------------------------------
    def __add__(self, other) -> Tensor:
        raise NotImplementedError

    def __radd__(self, other) -> Tensor:
        raise NotImplementedError

    def __neg__(self) -> Tensor:
        raise NotImplementedError

    def __sub__(self, other) -> Tensor:
        raise NotImplementedError

    def __rsub__(self, other) -> Tensor:
        raise NotImplementedError

    def __mul__(self, other) -> Tensor:
        raise NotImplementedError

    def __rmul__(self, other) -> Tensor:
        raise NotImplementedError

    def __truediv__(self, other) -> Tensor:
        raise NotImplementedError

    def __rtruediv__(self, other) -> Tensor:
        raise NotImplementedError

    def __pow__(self, exponent: float) -> Tensor:
        """Power with a constant (Python number) exponent."""
        raise NotImplementedError

    # ---- linear algebra -----------------------------------------------------------------------
    def __matmul__(self, other) -> Tensor:
        """Matrix product with numpy semantics: 2D @ 2D, and batched (..., n, k) @ (k, m) or
        (..., n, k) @ (..., k, m) with broadcasting over the leading batch dims. 1D operands are not
        required."""
        raise NotImplementedError

    def __rmatmul__(self, other) -> Tensor:
        raise NotImplementedError

    # ---- unary functions ------------------------------------------------------------------------
    def exp(self) -> Tensor:
        raise NotImplementedError

    def log(self) -> Tensor:
        raise NotImplementedError

    def relu(self) -> Tensor:
        raise NotImplementedError

    def tanh(self) -> Tensor:
        raise NotImplementedError

    def sigmoid(self) -> Tensor:
        """Numerically stable: must not overflow / warn for inputs like +-1000."""
        raise NotImplementedError

    # ---- reductions & shape ops ---------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        """axis: None, int (negative allowed) or tuple of ints."""
        raise NotImplementedError

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        raise NotImplementedError

    def reshape(self, *shape) -> Tensor:
        """Accepts reshape(2, 3) and reshape((2, 3)); -1 allowed."""
        raise NotImplementedError

    def transpose(self, *axes) -> Tensor:
        """No axes -> reverse all axes (like numpy). Otherwise a permutation."""
        raise NotImplementedError

    @property
    def T(self) -> Tensor:
        raise NotImplementedError

    def __getitem__(self, idx) -> Tensor:
        """Basic slicing AND integer-array (fancy) indexing, e.g. ``logits[np.arange(n), y]``.
        Fancy indices may repeat -- the backward pass must accumulate (hint: np.add.at)."""
        raise NotImplementedError

    def log_softmax(self, axis: int = -1) -> Tensor:
        """log(softmax(x)) along ``axis``. Must be stable for inputs like [1000, 0, -1000]."""
        raise NotImplementedError

    # ---- autodiff -------------------------------------------------------------------------------
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Reverse-mode sweep from this tensor through the graph (iterative topological order)."""
        raise NotImplementedError
