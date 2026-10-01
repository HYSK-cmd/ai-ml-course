"""Week 6 autograder: metrics exposition + drift statistics."""
import math
import re
import threading

import numpy as np
import pytest
from scipy.stats import ks_2samp

from serve.drift import DriftMonitor, ks_statistic, psi
from serve.metrics import Counter, Histogram, Registry


def parse_buckets(text, name):
    out = {}
    for le, v in re.findall(rf'^{name}_bucket\{{le="([^"]+)"\}} (\S+)$', text, re.M):
        out[math.inf if le == "+Inf" else float(le)] = float(v)
    return out


def test_counter():
    c = Counter("http_requests_total", "Requests served")
    c.inc()
    c.inc(2)
    assert c.value == 3
    text = c.render()
    assert "# TYPE http_requests_total counter" in text
    assert re.search(r"^http_requests_total 3(\.0)?$", text, re.M)


def test_histogram_exposition_is_cumulative():
    h = Histogram("lat", "Latency", [0.1, 0.5, 1.0])
    for v in [0.05, 0.1, 0.3, 0.7, 2.0, 5.0]:
        h.observe(v)
    text = h.render()
    assert "# TYPE lat histogram" in text and "# HELP lat Latency" in text
    b = parse_buckets(text, "lat")
    assert b == {0.1: 2, 0.5: 3, 1.0: 4, math.inf: 6}, "le is inclusive and counts are cumulative"
    assert re.search(r"^lat_count 6$", text, re.M)
    assert float(re.search(r"^lat_sum (\S+)$", text, re.M).group(1)) == pytest.approx(8.15)
    assert h.count == 6 and h.sum == pytest.approx(8.15)


def test_histogram_quantile_interpolation():
    h = Histogram("x", "x", [1, 2, 4])
    assert math.isnan(h.quantile(0.5))
    for v in [0.5] * 10 + [1.5] * 10:
        h.observe(v)
    assert h.quantile(0.5) == pytest.approx(1.0)   # rank 10 = end of bucket (0,1]
    assert h.quantile(0.75) == pytest.approx(1.5)  # halfway through (1,2]
    assert h.quantile(0.25) == pytest.approx(0.5)
    h.observe(100)
    assert h.quantile(1.0) == 4  # +Inf bucket -> largest finite bound


def test_thread_safety():
    c = Counter("c", "c")
    h = Histogram("h", "h", [1])

    def work():
        for _ in range(10000):
            c.inc()
            h.observe(0.5)
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert c.value == 80000 and h.count == 80000


def test_registry():
    r = Registry()
    c = r.register(Counter("a_total", "A"))
    r.register(Histogram("b", "B", [1]))
    with pytest.raises(ValueError):
        r.register(Counter("a_total", "dup"))
    c.inc()
    text = r.render()
    assert text.endswith("\n") and "a_total 1" in text and 'b_bucket{le="+Inf"} 0' in text


# ------------------------------------------------------------------------------ drift
def test_psi_hand_computed_example():
    expected = np.arange(100, dtype=float)
    actual = np.array([0.0] * 50 + [99.0] * 50)
    e, a = 0.25, np.array([0.5, 1e-6, 1e-6, 0.5])
    want = float(((a - e) * np.log(a / e)).sum())
    assert psi(expected, actual, bins=4) == pytest.approx(want, rel=1e-9)


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    assert psi(ref, rng.normal(size=20000)) < 0.01
    moderate = psi(ref, rng.normal(0.3, 1, size=20000))
    shifted = psi(ref, rng.normal(1.0, 1, size=20000))
    assert 0.05 < moderate < 0.25 < shifted


def test_ks_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(5):
        a = rng.normal(size=rng.integers(50, 500))
        b = rng.normal(0.2, 1.3, size=rng.integers(50, 500))
        assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic, abs=1e-12)
    ties = np.array([1.0, 1, 1, 2, 2, 3])
    assert ks_statistic(ties, np.array([1.0, 2, 3])) == pytest.approx(ks_2samp(ties, [1.0, 2, 3]).statistic)


def test_drift_monitor():
    rng = np.random.default_rng(2)
    m = DriftMonitor(rng.normal(size=5000), window=500, min_samples=100)
    m.update(rng.normal(size=50))
    r = m.report()
    assert math.isnan(r["psi"]) and r["drifted"] is False
    m.update(rng.normal(size=500))
    assert m.report()["drifted"] is False
    m.update(rng.normal(2.0, 1, size=500))  # the window now holds only shifted data
    r = m.report()
    assert r["drifted"] is True and r["ks"] > 0.5
