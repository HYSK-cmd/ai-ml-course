# ai-ml-course

Hoon's 8-week, build-everything-from-scratch AI/ML engineering course. Every week you implement one
layer of a single stack, and each layer is used by the next:

| Week | Topic | You build | Used later by |
|------|-------|-----------|---------------|
| 1 | Math for ML | `forge.autograd`, `forge.optim`, `forge.gradcheck` — reverse-mode autodiff over numpy | everything in W2–W3 |
| 2 | Machine Learning | `forge.ml` — linear/logistic regression, CART, gradient boosting, k-means, GMM/EM, PCA, metrics, CV | W8 ranker, IVF index |
| 3 | Deep Learning I | `forge.nn` — Module system, Linear, BatchNorm, LayerNorm, Conv2d, MaxPool, Dropout; MNIST CNN | W5 perf work |
| 4 | Deep Learning II | `forge.gpt` — BPE tokenizer, GPT with KV cache, AMP training loop (PyTorch, GPU) | W5 CLI, W6 server |
| 5 | Python Engineering | `pyeng.drills`, `forge.cli`, packaging, mypy --strict, ruff, CI, profiling | the rest of your career |
| 6 | MLOps | `serve` — dynamic batcher, FastAPI + SSE, Prometheus metrics, drift, MLflow, Docker, DVC, load test | W7, W8 |
| 7 | System Design & HLD | `sysdesign` — consistent hashing, Bloom filter, LSM-tree KV store, rate limiters; an HLD doc | W8 service |
| 8 | ML System Design | `recsys` — MovieLens two-tower retrieval, IVF ANN, GBM ranker, A/B stats, recsys service; an ML design doc | — |

Lessons, derivations, quizzes, hints and the Claude tutor live in the course web app (link in your notes).
This repo holds **stubs + an autograder**. Nothing is implemented for you except plumbing
(`scripts/get_data.py`, `benchmarks/bench_conv.py`).

## Setup (already done on this machine)

```powershell
cd ~/Desktop/ai-ml-course
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,w4,w6]"   # torch: pip install torch --index-url https://download.pytorch.org/whl/cu126
python scripts/get_data.py       # MNIST, TinyShakespeare, California housing, MovieLens-1M -> data/
```

The `.venv` here already has numpy, pytest, hypothesis, scikit-learn, scipy, pandas, mypy, ruff, CUDA PyTorch,
FastAPI and MLflow, and `data/` is downloaded. Activate it and go. (`pip install -e .` is a Week 5 task.)

## Running the grader

```powershell
pytest tests/w1 -q                 # one week, fast tests
pytest tests/w1 -q -k unbroadcast  # one milestone
pytest tests/w3 -m slow -s         # training/accuracy tests (MNIST, GPT, recommender)
pytest tests/w2 -m stretch         # stretch goals
pytest -q                          # everything fast
```

Every test fails with `NotImplementedError` until you implement it. The tests are the spec: read a
test when a docstring is ambiguous.

Reference timings on this machine (RTX 3060, for a solid solution): W3 CNN 3 epochs ≈ 1 min (numpy, CPU),
W4 Shakespeare GPT 2,500 steps ≈ 6 min (val loss ≈ 1.50), W8 pipeline ≈ 1 min.

## Rules

1. Write every line of project code yourself. Use AI to explain concepts, not to write your implementation.
2. Derive before you code. If you can't derive a backward pass on paper, you can't debug it.
3. Commit after each milestone: `git commit -am "w1: broadcasting"`. Your history is your portfolio.
4. Deliverable docs go in `docs/` (templates in `docs/templates/`).
