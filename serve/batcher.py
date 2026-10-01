"""Dynamic batching: the single biggest throughput lever in model serving.

Callers ``await batcher.submit(item)`` concurrently. A background task collects pending items and
calls ``process_batch(items) -> results`` (same length, same order) when EITHER max_batch_size items
are waiting OR the oldest waiting item has waited max_wait_ms. Each caller gets its own result.

Rules:
* process_batch may be a coroutine function (await it) or a plain function (run it with
  asyncio.to_thread so GPU work doesn't block the event loop).
* If process_batch raises, every caller in THAT batch gets the exception; the batcher keeps serving.
* ``start()`` launches the background task; ``stop()`` finishes all already-submitted items, then
  ends the task. ``submit`` after ``stop`` raises RuntimeError.
* ``batch_sizes``: list of the sizes of every batch processed (for metrics and the tests).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class DynamicBatcher:
    def __init__(self, process_batch: Callable[[list[Any]], list[Any] | Awaitable[list[Any]]],
                 max_batch_size: int = 8, max_wait_ms: float = 5.0):
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def submit(self, item: Any) -> Any:
        raise NotImplementedError
