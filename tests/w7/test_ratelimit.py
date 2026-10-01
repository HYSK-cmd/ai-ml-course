"""Week 7 autograder: rate limiters, caching (stretch), and the limiter wired into the W6 server."""
import math
import threading
import time
from pathlib import Path

import pytest

from sysdesign.cache import SingleFlight, TTLCache
from sysdesign.ratelimit import KeyedLimiter, SlidingWindowCounter, SlidingWindowLog, TokenBucket

ROOT = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_token_bucket_burst_and_refill():
    c = Clock()
    b = TokenBucket(rate=2, capacity=5, clock=c)
    assert [b.allow() for _ in range(6)] == [True] * 5 + [False]
    assert b.retry_after() == pytest.approx(0.5)
    c.t += 0.5
    assert b.allow() and not b.allow()
    c.t += 100  # long idle: refill caps at capacity
    assert sum(b.allow() for _ in range(10)) == 5
    assert b.allow(cost=0) and b.retry_after(cost=0) == 0.0
    c.t += 1.5
    assert b.allow(cost=3) and not b.allow(cost=1)
    assert TokenBucket(0, 1, clock=c).retry_after(2) == math.inf


def test_sliding_window_log_prevents_boundary_burst():
    c = Clock(0.0)
    lim = SlidingWindowLog(limit=10, window=1.0, clock=c)
    c.t = 0.9
    assert sum(lim.allow() for _ in range(12)) == 10
    c.t = 1.1  # a fixed-window limiter would allow 10 more here (20 in 0.2s)
    assert not lim.allow()
    c.t = 1.95
    assert sum(lim.allow() for _ in range(12)) == 10


def test_sliding_window_counter_estimate():
    c = Clock(0.0)
    lim = SlidingWindowCounter(limit=10, window=1.0, clock=c)
    c.t = 0.9
    assert sum(lim.allow() for _ in range(12)) == 10
    c.t = 1.1  # estimate = 10 * (1 - 0.1) + 0 = 9 -> exactly one more allowed
    assert sum(lim.allow() for _ in range(5)) == 1
    c.t = 1.5  # estimate = 10 * 0.5 + 1 = 6 -> 4 more
    assert sum(lim.allow() for _ in range(10)) == 4
    c.t = 3.2  # previous window (2,3) was empty -> fresh
    assert sum(lim.allow() for _ in range(12)) == 10


def test_thread_safety():
    b = TokenBucket(rate=0, capacity=1000)
    counts = []

    def hammer():
        counts.append(sum(b.allow() for _ in range(500)))
    ts = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(counts) == 1000


def test_keyed_limiter_isolation_and_lru():
    c = Clock()
    kl = KeyedLimiter(lambda: TokenBucket(0, 2, clock=c), max_keys=3)
    assert kl.allow("alice") and kl.allow("alice") and not kl.allow("alice")
    assert kl.allow("bob")
    kl.allow("carol")
    kl.allow("alice")  # touch alice so bob is the LRU
    kl.allow("dave")   # evicts bob
    assert len(kl) == 3
    assert kl.allow("bob") and kl.allow("bob"), "evicted keys start fresh"
    assert not kl.allow("alice"), "alice was not evicted (recently used)"


def test_server_rate_limiting():
    from fastapi.testclient import TestClient

    from serve.app import create_app

    class Echo:
        def generate_batch(self, reqs):
            return [r.prompt for r in reqs]

        def stream(self, req):
            yield req.prompt
    limiter = KeyedLimiter(lambda: TokenBucket(rate=0, capacity=3))
    with TestClient(create_app(Echo(), rate_limiter=limiter)) as client:
        codes = [client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "k1"}).status_code
                 for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        r = client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "k1"})
        assert "retry-after" in {h.lower() for h in r.headers}
        assert client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "k2"}).status_code == 200
        assert all(client.get("/health").status_code == 200 for _ in range(10)), "/health is never limited"


def test_hld_doc():
    doc = ROOT / "docs" / "w7_url_shortener.md"
    assert doc.exists(), "write the HLD in docs/w7_url_shortener.md (template: docs/templates/hld.md)"
    text = doc.read_text(encoding="utf-8").lower()
    for section in ["requirements", "estimat", "api", "data model", "high-level", "deep dive", "trade-off"]:
        assert section in text, f"missing section about {section!r}"
    assert len(text.split()) >= 800


# ------------------------------------------------------------------------------ stretch: caching
@pytest.mark.stretch
def test_ttl_cache():
    c = Clock()
    cache = TTLCache(maxsize=2, ttl=10, clock=c)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # a is now most recently used
    cache.set("c", 3)           # evicts b
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    c.t += 11
    assert cache.get("a", "miss") == "miss"


@pytest.mark.stretch
def test_single_flight_dedupes_concurrent_calls():
    sf = SingleFlight()
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return 42
    results = []
    ts = [threading.Thread(target=lambda: results.append(sf.do("k", slow))) for _ in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results == [42] * 10 and len(calls) == 1
    assert sf.do("k", lambda: 7) == 7, "after completion, a new call runs again"
