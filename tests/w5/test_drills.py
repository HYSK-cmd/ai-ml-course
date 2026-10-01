"""Week 5 autograder: Python internals drills."""
import asyncio
import itertools
import math
import os
import time

import pytest

from pyeng.drills import Timer, Typed, Vector, chunked, gather_limited, parallel_map, retry, sliding_window


# ------------------------------------------------------------------------------- Vector
def test_vector_basics():
    v, w = Vector(3, 4), Vector(1, 2)
    assert repr(v) == "Vector(3.0, 4.0)"
    assert v + w == Vector(4, 6) and v - w == Vector(2, 2)
    assert v * 2 == Vector(6, 8) and 2 * v == Vector(6, 8)
    assert v @ w == 11.0
    assert abs(v) == 5.0
    assert bool(v) and not bool(Vector(0, 0))
    assert len(Vector(1, 2, 3)) == 3 and list(Vector(1, 2)) == [1.0, 2.0]
    assert Vector(1, 2, 3)[1] == 2.0 and Vector(1, 2, 3)[1:] == Vector(2, 3)
    assert isinstance(Vector(1, 2, 3)[1:], Vector)


def test_vector_errors_and_immutability():
    with pytest.raises(ValueError):
        Vector(1, 2) + Vector(1, 2, 3)
    with pytest.raises(TypeError):
        Vector(1, 2) * Vector(1, 2)
    v = Vector(1, 2)
    with pytest.raises(AttributeError):
        v.x = 5
    with pytest.raises(AttributeError):
        v.components = (9, 9)
    assert not hasattr(v, "__dict__"), "use __slots__"


def test_vector_hashable():
    d = {Vector(1, 2): "a"}
    assert d[Vector(1.0, 2.0)] == "a"
    assert len({Vector(1, 2), Vector(1, 2), Vector(2, 1)}) == 2
    assert Vector(1, 2) != (1.0, 2.0)


# ------------------------------------------------------------------------------- descriptor
def make_order_class():
    class Order:
        qty = Typed(int, lambda v: v > 0)
        name = Typed(str)
    return Order


def test_typed_descriptor():
    Order = make_order_class()
    a, b = Order(), Order()
    a.qty, a.name = 3, "gpu"
    b.qty = 7
    assert (a.qty, a.name, b.qty) == (3, "gpu", 7)
    with pytest.raises(TypeError):
        a.qty = "3"
    with pytest.raises(ValueError):
        a.qty = -1
    assert a.qty == 3, "failed assignments must not change the value"
    with pytest.raises(AttributeError):
        _ = b.name
    assert isinstance(Order.qty, Typed)


# ------------------------------------------------------------------------------- generators
def test_chunked():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked([], 3)) == []
    assert next(chunked(itertools.count(), 2)) == [0, 1], "must be lazy"
    with pytest.raises(ValueError):
        chunked(range(3), 0)  # raised at call time, before iteration


def test_sliding_window():
    assert list(sliding_window([1, 2, 3, 4], 2)) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window([1, 2], 3)) == []
    it = sliding_window(itertools.count(), 3)
    assert next(it) == (0, 1, 2) and next(it) == (1, 2, 3)


# ------------------------------------------------------------------------------- context manager
def test_timer():
    with Timer() as t:
        time.sleep(0.05)
    assert 0.04 < t.elapsed < 0.5
    with pytest.raises(KeyError):
        with Timer() as t2:
            raise KeyError("boom")
    assert t2.elapsed >= 0


# ------------------------------------------------------------------------------- decorator
def test_retry():
    sleeps, calls = [], []

    @retry(times=4, exceptions=(ConnectionError,), backoff=0.1, sleep=sleeps.append)
    def flaky(x):
        """doc"""
        calls.append(x)
        if len(calls) < 3:
            raise ConnectionError
        return x * 2
    assert flaky(21) == 42 and len(calls) == 3
    assert sleeps == [0.1, 0.2]
    assert flaky.__name__ == "flaky" and flaky.__doc__ == "doc" and hasattr(flaky, "__wrapped__")

    @retry(times=2, exceptions=(ConnectionError,), sleep=lambda s: None)
    def always():
        raise ConnectionError("down")
    with pytest.raises(ConnectionError, match="down"):
        always()

    n = []

    @retry(times=5, exceptions=(ConnectionError,), sleep=lambda s: None)
    def wrong_kind():
        n.append(1)
        raise ValueError
    with pytest.raises(ValueError):
        wrong_kind()
    assert len(n) == 1, "non-listed exceptions are not retried"


# ------------------------------------------------------------------------------- concurrency
def test_gather_limited_bounds_concurrency_and_keeps_order():
    state = {"now": 0, "peak": 0}

    def make(i):
        async def job():
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01 * (5 - i % 5))
            state["now"] -= 1
            return i * i
        return job
    t = time.perf_counter()
    out = asyncio.run(gather_limited([make(i) for i in range(20)], limit=4))
    assert out == [i * i for i in range(20)]
    assert state["peak"] == 4
    assert time.perf_counter() - t < 0.5, "jobs must actually run concurrently"


class Boom(Exception):
    pass


def test_gather_limited_propagates_errors():
    async def ok():
        return 1

    async def bad():
        raise Boom("x")
    with pytest.raises(Boom):
        asyncio.run(gather_limited([ok, bad, ok], limit=2))


def test_parallel_map_uses_processes():
    assert parallel_map(math.factorial, range(10), workers=2) == [math.factorial(i) for i in range(10)]
    pids = set(parallel_map(_pid, range(8), workers=2))
    assert os.getpid() not in pids, "work must run in child processes"


def _pid(_):
    return os.getpid()
