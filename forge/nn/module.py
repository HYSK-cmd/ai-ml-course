"""Module system. The key trick is ``__setattr__``: assigning a Parameter or Module to an attribute
registers it, so ``parameters()`` can find everything recursively. (Week 5 revisits this as a
lesson on Python's data model.)"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from forge.autograd import Tensor


class Parameter(Tensor):
    """A Tensor that is always a trainable leaf (requires_grad=True)."""

    def __init__(self, data):
        raise NotImplementedError


class Module:
    """Contract:
    * ``self.x = Parameter(...)`` / ``self.x = SomeModule(...)`` registers it under name "x" (in
      assignment order). Re-assigning a name replaces the entry.
    * ``register_buffer(name, array)``: non-trainable state saved in state_dict (e.g. BatchNorm
      running stats). Readable as ``self.name``; assigning a new ndarray to ``self.name`` updates it.
    * ``named_parameters()``: list of (dotted_name, Parameter) -- own params first, then children
      recursively, e.g. "0.weight", "encoder.fc.bias". The same Parameter object appears once
      (weight sharing!).
    * ``state_dict()``: {dotted_name: COPY of ndarray} for params and buffers.
      ``load_state_dict(sd)``: copies values in (in place); raise KeyError on missing/unexpected keys
      and ValueError on shape mismatch.
    * ``train()`` / ``eval()`` set ``.training`` on this module and all descendants, return self.
    * ``to(dtype)`` casts all params and buffers in place (``p.data = p.data.astype(dtype)``), returns self.
    * ``__call__`` forwards to ``forward``.
    Subclasses call ``super().__init__()`` first.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        """Only called when normal lookup fails -- use it to expose buffers."""
        raise NotImplementedError

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def children(self) -> Iterator[Module]:
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        raise NotImplementedError

    def named_buffers(self, prefix: str = "") -> list[tuple[str, np.ndarray]]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    def train(self, mode: bool = True) -> Module:
        raise NotImplementedError

    def eval(self) -> Module:
        raise NotImplementedError

    def to(self, dtype) -> Module:
        raise NotImplementedError

    def state_dict(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class Sequential(Module):
    """Sequential(m0, m1, ...) registers children under names "0", "1", ...; supports len() and [i]."""

    def __init__(self, *modules: Module):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> Module:
        raise NotImplementedError
