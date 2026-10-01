# Week 5 performance write-up

> Copy to `docs/w5_perf.md`. 150+ words. Evidence over opinions.

## Baseline
`python benchmarks/bench_conv.py --record baseline` -> ___ ms

## What the profiler showed
Paste the top of `python -m cProfile -s tottime benchmarks/bench_conv.py`. Which function dominates?
Is it Python overhead (many small calls) or numpy work (few big calls)? Memory layout / copies?

## Changes (one at a time, re-measured after each)
| change | ms before | ms after | why it helped |
|--------|-----------|----------|---------------|

## Result
`--record optimized` -> ___ ms (___x). What would you try next (numba, float32 everywhere, fusing ops)?
