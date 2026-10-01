"""Week 8 slow tests: the two-tower model and the full pipeline on MovieLens-1M (GPU recommended).
    pytest tests/w8 -m slow -s        (~1-2 min on an RTX 3060 for the reference solution)"""
import numpy as np
import pytest
import torch

from recsys.pipeline import PipelineConfig, build, evaluate
from tests.helpers import need_data

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def built():
    need_data("ml-1m/ratings.dat")
    rec, ctx = build(PipelineConfig())
    return rec, ctx, evaluate(rec, ctx)


def test_embeddings_are_normalized(built):
    _, ctx, _ = built
    U, I = ctx["user_vecs"], ctx["item_vecs"]
    assert U.shape[1] == I.shape[1] and U.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(I, axis=1), 1, atol=1e-4)


def test_beats_popularity(built):
    _, _, r = built
    print({k: round(v, 4) for k, v in r.items()})
    assert r["recall@50"] > 1.3 * r["pop_recall@50"], "retrieval should clearly beat popularity (is logQ correction on?)"
    assert r["ndcg@10"] > 1.05 * r["pop_ndcg@10"]
    assert r["coverage@10"] > 5 * r["pop_coverage@10"], "a personalized model recommends far more of the catalog"


def test_recommend_api(built):
    rec, ctx, _ = built
    raw_user = next(iter(ctx["user_index"]))
    items, fallback = rec.recommend(raw_user, 10)
    assert not fallback and len(items) == 10 and len(set(items)) == 10
    seen_raw = set(ctx["train"].loc[ctx["train"]["user_id"] == raw_user, "item_id"])
    assert not (set(items) & seen_raw), "never recommend what the user already interacted with"
    cold, fb = rec.recommend(10**9, 5)
    assert fb and len(cold) == 5


@pytest.mark.stretch
def test_ranker_improves_ndcg():
    need_data("ml-1m/ratings.dat")
    rec, ctx = build(PipelineConfig(use_ranker=True))
    r = evaluate(rec, ctx)
    print({k: round(v, 4) for k, v in r.items()})
    assert r["ndcg@10"] > 1.15 * r["pop_ndcg@10"]
