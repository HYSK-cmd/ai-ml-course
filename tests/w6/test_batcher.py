"""Week 6 autograder: DynamicBatcher (asyncio)."""
import asyncio
import threading
import time

import pytest

from serve.batcher import DynamicBatcher


def run(coro):
    return asyncio.run(coro)


def test_results_routed_to_the_right_caller():
    async def main():
        b = DynamicBatcher(lambda xs: [x * 10 for x in xs], max_batch_size=8, max_wait_ms=5)
        await b.start()
        out = await asyncio.gather(*(b.submit(i) for i in range(100)))
        await b.stop()
        return out, b.batch_sizes
    out, sizes = run(main())
    assert out == [i * 10 for i in range(100)]
    assert sum(sizes) == 100
    assert max(sizes) <= 8
    assert sum(sizes) / len(sizes) > 4, f"a burst of 100 should form mostly full batches, got {sizes}"


def test_async_process_fn_and_shuffled_completion():
    async def process(xs):
        await asyncio.sleep(0.01)
        return [x.upper() for x in xs]

    async def main():
        b = DynamicBatcher(process, max_batch_size=4, max_wait_ms=2)
        await b.start()

        async def client(i):
            await asyncio.sleep((i * 7 % 5) / 1000)
            return await b.submit(f"req{i}")
        out = await asyncio.gather(*(client(i) for i in range(30)))
        await b.stop()
        return out
    assert run(main()) == [f"REQ{i}" for i in range(30)]


def test_sync_process_fn_runs_off_the_event_loop():
    loop_thread = {}

    def process(xs):
        loop_thread["t"] = threading.get_ident()
        time.sleep(0.05)  # simulated GPU work
        return xs

    async def main():
        b = DynamicBatcher(process, max_batch_size=2, max_wait_ms=1)
        await b.start()
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.005)
                ticks += 1
        await asyncio.gather(b.submit(1), ticker())
        await b.stop()
        return ticks, threading.get_ident()
    ticks, main_thread = run(main())
    assert loop_thread["t"] != main_thread, "sync process_batch must run in a worker thread"
    assert ticks >= 5, "the event loop was blocked while the batch ran"


def test_lone_request_flushes_after_max_wait():
    async def main():
        b = DynamicBatcher(lambda xs: xs, max_batch_size=64, max_wait_ms=30)
        await b.start()
        t = time.perf_counter()
        await b.submit("x")
        dt = time.perf_counter() - t
        await b.stop()
        return dt
    dt = run(main())
    assert 0.025 <= dt < 0.25, f"a single request should wait ~max_wait_ms (30ms), took {dt * 1000:.0f}ms"


def test_full_batch_does_not_wait():
    async def main():
        b = DynamicBatcher(lambda xs: xs, max_batch_size=4, max_wait_ms=1000)
        await b.start()
        t = time.perf_counter()
        await asyncio.gather(*(b.submit(i) for i in range(4)))
        dt = time.perf_counter() - t
        await b.stop()
        return dt
    assert run(main()) < 0.3, "a full batch must be processed immediately, not after max_wait_ms"


def test_errors_propagate_to_that_batch_only():
    calls = {"n": 0}

    def process(xs):
        calls["n"] += 1
        if "boom" in xs:
            raise ValueError("bad batch")
        return xs

    async def main():
        b = DynamicBatcher(process, max_batch_size=2, max_wait_ms=1)
        await b.start()
        r1 = await asyncio.gather(b.submit("a"), b.submit("boom"), return_exceptions=True)
        r2 = await b.submit("ok")  # the batcher must still be alive
        await b.stop()
        return r1, r2
    r1, r2 = run(main())
    assert all(isinstance(r, ValueError) for r in r1)
    assert r2 == "ok"


def test_stop_drains_and_rejects_new():
    async def main():
        b = DynamicBatcher(lambda xs: xs, max_batch_size=2, max_wait_ms=20)
        await b.start()
        tasks = [asyncio.create_task(b.submit(i)) for i in range(5)]
        await asyncio.sleep(0)  # let them enqueue
        await b.stop()
        done = [t.result() for t in tasks]
        with pytest.raises(RuntimeError):
            await b.submit(99)
        return done
    assert run(main()) == [0, 1, 2, 3, 4]
