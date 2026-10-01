"""Week 5 drills: the Python machinery that frameworks (PyTorch, FastAPI, pydantic, pytest) are built on.
Each one maps to a lesson. No third-party libraries."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from typing import Any, Self, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------- data model
class Vector:
    """Immutable n-dimensional vector of floats.

    Vector(1, 2)                -> repr "Vector(1.0, 2.0)"
    v + w, v - w                -> Vector (ValueError if lengths differ)
    v * 3, 3 * v                -> Vector (scalar only; Vector * Vector -> TypeError via NotImplemented)
    v @ w                       -> float dot product
    abs(v)                      -> Euclidean norm;  bool(v) False iff all zeros
    len(v), iter(v), v[i] -> float, v[1:3] -> Vector
    ==, hash                    -> by value; usable as dict keys
    v.x = 1 / v.anything = 1    -> AttributeError (immutable; hint: __slots__ and __setattr__)
    """

    def __init__(self, *components: float):
        raise NotImplementedError


# --------------------------------------------------------------------------- descriptors
class Typed:
    """Validating descriptor -- the mechanism behind dataclass fields, Django/pydantic models and
    forge.nn.Module's attribute magic.

        class Order:
            qty = Typed(int, lambda v: v > 0)
            name = Typed(str)

    * assigning a value of the wrong type raises TypeError; a failing validator raises ValueError
    * reading an attribute never assigned raises AttributeError
    * values are stored per INSTANCE (two Orders don't share qty); use __set_name__ to learn the name
    * accessing on the class (Order.qty) returns the descriptor itself
    """

    def __init__(self, type_: type, validator: Callable[[Any], bool] | None = None):
        raise NotImplementedError


# --------------------------------------------------------------------------- iterators & generators
def chunked(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Lazily yield lists of n items (last may be shorter). Must work on infinite iterators.
    n < 1 -> ValueError (raised when called, not on first next() -- think about why that's tricky)."""
    raise NotImplementedError


def sliding_window(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """(1,2,3,4), n=2 -> (1,2), (2,3), (3,4). Lazy; O(n) memory (collections.deque)."""
    raise NotImplementedError


# --------------------------------------------------------------------------- context managers
class Timer:
    """with Timer() as t: ...  -> t.elapsed (seconds, float) set on exit. Must NOT swallow exceptions
    (elapsed is still recorded). Uses time.perf_counter."""

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        raise NotImplementedError


# --------------------------------------------------------------------------- decorators
def retry(times: int = 3, exceptions: tuple[type[BaseException], ...] = (Exception,), backoff: float = 0.0,
          sleep: Callable[[float], None] = time.sleep) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator factory. Call the function up to ``times`` times while it raises one of
    ``exceptions``; other exceptions propagate immediately. Between attempts sleep
    backoff * 2**attempt_index (0, 1, ...). After the last failure re-raise that exception.
    The wrapper must keep __name__, __doc__ and __wrapped__ (functools.wraps)."""
    raise NotImplementedError


# --------------------------------------------------------------------------- concurrency
async def gather_limited(factories: Iterable[Callable[[], Awaitable[R]]], limit: int) -> list[R]:
    """Run coroutine factories with at most ``limit`` in flight at once; return results in INPUT order.
    If any raises, propagate the exception. (asyncio.Semaphore -- and know why a plain gather with
    10k coroutines hitting one server is a bad idea.)"""
    raise NotImplementedError


def parallel_map(fn: Callable[[T], R], items: Iterable[T], workers: int = 4) -> list[R]:
    """Map fn over items in a PROCESS pool (CPU-bound work escapes the GIL); results in input order.
    fn must be picklable (module-level)."""
    raise NotImplementedError
