"""Week 5 benchmark harness (provided). Times forward+backward of a conv block with YOUR forge.nn.

    python benchmarks/bench_conv.py                     # just measure
    python benchmarks/bench_conv.py --record baseline   # save as your W3 baseline (do this FIRST)
    python benchmarks/bench_conv.py --record optimized  # after optimizing

Profile before optimizing:
    python -m cProfile -s cumtime benchmarks/bench_conv.py | head -30
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).with_name("results.json")


def measure(repeats: int = 5) -> float:
    """Median milliseconds for Conv2d(16->32, 3x3, pad 1) + ReLU + MaxPool2d(2), fwd+bwd, batch 64x28x28."""
    from forge.autograd import Tensor
    from forge.nn import Conv2d, MaxPool2d, ReLU, Sequential

    rng = np.random.default_rng(0)
    block = Sequential(Conv2d(16, 32, 3, padding=1, rng=rng), ReLU(), MaxPool2d(2))
    x = rng.normal(size=(64, 16, 28, 28)).astype(np.float32)
    times = []
    for _ in range(repeats + 1):
        xt = Tensor(x, requires_grad=True)
        t = time.perf_counter()
        block(xt).sum().backward()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times[1:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", choices=["baseline", "optimized"])
    args = ap.parse_args()
    ms = measure()
    print(f"conv block fwd+bwd: {ms:.1f} ms")
    if args.record:
        data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        data[f"{args.record}_ms"] = ms
        RESULTS.write_text(json.dumps(data, indent=2))
        print(f"recorded {args.record} -> {RESULTS}")
