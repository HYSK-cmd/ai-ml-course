"""Week 8 autograder: data splitting, metrics, baseline, ANN index, A/B statistics, service."""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.abtest import sample_size_per_arm, simulate_power, two_proportion_ztest
from recsys.ann import IVFIndex, brute_force_search
from recsys.baselines import PopularityRecommender
from recsys.data import encode_ids, implicit, load_movies, load_ratings, temporal_split
from recsys.metrics import catalog_coverage, hit_rate_at_k, ndcg_at_k, recall_at_k
from tests.helpers import need_data

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------ data
@pytest.fixture(scope="module")
def ml():
    need_data("ml-1m/ratings.dat")
    return load_ratings(str(ROOT / "data" / "ml-1m")), load_movies(str(ROOT / "data" / "ml-1m"))


def test_loaders(ml):
    ratings, movies = ml
    assert len(ratings) == 1_000_209 and list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert ratings["user_id"].nunique() == 6040
    assert (ratings.dtypes == "int64").all()
    toy = movies[movies["item_id"] == 1].iloc[0]
    assert toy["title"] == "Toy Story (1995)" and toy["genres"] == ["Animation", "Children's", "Comedy"]
    assert len(implicit(ratings)) == int((ratings["rating"] >= 4).sum())


def test_temporal_split_no_leakage(ml):
    ratings, _ = ml
    pos = implicit(ratings)
    train, test = temporal_split(pos, test_frac=0.2, min_train=5)
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    common = first_test.index
    assert (last_train[common] <= first_test).all(), "a user's test interactions must come after their train ones"
    assert set(test["item_id"]) <= set(train["item_id"])
    assert len(train) + len(test) <= len(pos)
    assert 0.17 < len(test) / len(pos) < 0.21
    small = pos.groupby("user_id").size()
    tiny_users = small[small < 6].index
    assert not test["user_id"].isin(tiny_users).any()


def test_temporal_split_toy_exact():
    df = pd.DataFrame({"user_id": [1] * 6 + [2] * 3, "item_id": [10, 11, 12, 13, 14, 15, 10, 11, 12],
                       "rating": [5] * 9, "timestamp": [6, 5, 4, 3, 2, 1, 1, 2, 3]})
    train, test = temporal_split(df, test_frac=0.2, min_train=5)
    assert sorted(test["item_id"]) == [10, 11]  # user 1's two latest (ceil(0.2*6)=2); user 2 too small
    assert len(train) == 7


def test_encode_ids_from_train_only():
    train = pd.DataFrame({"user_id": [7, 7, 9], "item_id": [100, 200, 100], "timestamp": [1, 2, 3]})
    test = pd.DataFrame({"user_id": [7, 9, 8], "item_id": [300, 200, 100], "timestamp": [4, 5, 6]})
    tr, te, ui, ii = encode_ids(train, test)
    assert ui == {7: 0, 9: 1} and ii == {100: 0, 200: 1}
    assert te[["u", "i"]].values.tolist() == [[1, 1]]  # unknown item 300 and unknown user 8 dropped
    assert "u" not in train.columns, "don't mutate the caller's frames"


# ------------------------------------------------------------------------------ metrics
def test_metrics_hand_computed():
    recs = {"a": [1, 2, 3, 4], "b": [9, 8, 7, 6], "c": [5]}
    truth = {"a": {2, 4, 10}, "b": {6}, "c": set(), "d": {1}}
    # users with non-empty truth: a, b, d (d has no recs)
    assert recall_at_k(recs, truth, 4) == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert recall_at_k(recs, truth, 2) == pytest.approx((1 / 3 + 0 + 0) / 3)
    assert hit_rate_at_k(recs, truth, 4) == pytest.approx(2 / 3)
    ndcg_a = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    ndcg_b = (1 / math.log2(5)) / 1.0
    assert ndcg_at_k(recs, truth, 4) == pytest.approx((ndcg_a + ndcg_b + 0) / 3)
    assert catalog_coverage(recs, n_items=20, k=2) == pytest.approx(5 / 20)


def test_popularity_baseline():
    train = pd.DataFrame({"i": [3, 3, 3, 1, 1, 2, 2, 5]})
    pop = PopularityRecommender().fit(train)
    assert pop.recommend(3) == [3, 1, 2]  # 1 and 2 tie at 2 -> smaller id first
    assert pop.recommend(3, exclude={3}) == [1, 2, 5]


# ------------------------------------------------------------------------------ ANN
def clustered(n, d=64, centers=200, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(centers, d))
    X = C[rng.integers(0, centers, n)] + 0.35 * rng.normal(size=(n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_brute_force_exact():
    X = clustered(2000)
    Q = clustered(20, seed=1)
    s, ids = brute_force_search(X, Q, 5)
    full = Q @ X.T
    np.testing.assert_array_equal(ids, np.argsort(-full, axis=1)[:, :5])
    np.testing.assert_allclose(s, np.sort(full, axis=1)[:, ::-1][:, :5], rtol=1e-5)


def test_ivf_recall_and_speed():
    X = clustered(200_000)
    Q = clustered(200, seed=1)
    idx = IVFIndex(n_lists=512, n_probe=16, seed=0)
    idx.train(X[:30_000])
    idx.add(X, ids=np.arange(200_000) + 7)  # custom ids must be returned
    _, true_ids = brute_force_search(X, Q, 10)
    s, ids = idx.search(Q, 10)
    recall = np.mean([len(set(a) & set(b + 7)) / 10 for a, b in zip(ids, true_ids)])
    assert recall >= 0.9, f"IVF recall@10 {recall:.3f}"
    assert np.all(np.diff(s, axis=1) <= 1e-6), "rows sorted by descending score"
    # single-query latency: the serving scenario
    t = time.perf_counter()
    for q in Q[:100]:
        brute_force_search(X, q[None], 10)
    brute = time.perf_counter() - t
    t = time.perf_counter()
    for q in Q[:100]:
        idx.search(q[None], 10)
    ivf = time.perf_counter() - t
    assert ivf * 5 <= brute, f"IVF {ivf * 10:.2f} ms/query vs brute force {brute * 10:.2f} ms/query -- need >= 5x"


def test_ivf_pads_when_too_few_candidates():
    X = clustered(50)
    idx = IVFIndex(n_lists=10, n_probe=1)
    idx.train(X)
    idx.add(X)
    s, ids = idx.search(X[:3], 40)
    assert ids.shape == (3, 40)
    assert (ids == -1).any() and np.all(np.isneginf(s[ids == -1]))


# ------------------------------------------------------------------------------ A/B testing
def test_sample_size():
    assert sample_size_per_arm(0.10, 0.02) == 3841
    assert sample_size_per_arm(0.05, 0.005) == 31234
    assert sample_size_per_arm(0.3, 0.03, alpha=0.01, power=0.9) == 7133


def test_ztest():
    z, p = two_proportion_ztest(200, 2000, 250, 2000)
    assert z == pytest.approx(2.501955, rel=1e-5) and p == pytest.approx(0.0123509, rel=1e-4)
    z, p = two_proportion_ztest(250, 2000, 200, 2000)
    assert z < 0 and p == pytest.approx(0.0123509, rel=1e-4)


def test_simulated_power_matches_design():
    n = sample_size_per_arm(0.10, 0.02)
    assert abs(simulate_power(0.10, 0.12, n, n_sims=4000, seed=1) - 0.80) < 0.03
    assert abs(simulate_power(0.10, 0.10, n, n_sims=4000, seed=2) - 0.05) < 0.015, "false-positive rate = alpha"


# ------------------------------------------------------------------------------ service
class FakeRec:
    def recommend(self, raw_user_id, k=10):
        if raw_user_id == 999999:
            return [1, 2, 3][:k], True
        return [raw_user_id + j for j in range(k)], False


def test_recsys_service():
    from fastapi.testclient import TestClient

    from recsys.service import create_recsys_app
    from sysdesign.ratelimit import KeyedLimiter, TokenBucket
    with TestClient(create_recsys_app(FakeRec())) as c:
        assert c.get("/health").json() == {"status": "ok"}
        r = c.get("/recommend/10", params={"k": 3}).json()
        assert r == {"user_id": 10, "items": [10, 11, 12], "fallback": False}
        assert c.get("/recommend/999999").json()["fallback"] is True
        assert c.get("/recommend/10", params={"k": 0}).status_code == 422
        assert c.get("/recommend/10", params={"k": 101}).status_code == 422
        m = c.get("/metrics").text
        assert "recommend_latency_seconds_count" in m and "recommend_fallback_total 1" in m
    lim = KeyedLimiter(lambda: TokenBucket(0, 2))
    with TestClient(create_recsys_app(FakeRec(), rate_limiter=lim)) as c:
        codes = [c.get("/recommend/1").status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_ml_design_doc():
    doc = ROOT / "docs" / "w8_ml_design.md"
    assert doc.exists(), "write docs/w8_ml_design.md (template: docs/templates/ml_design.md)"
    text = doc.read_text(encoding="utf-8").lower()
    for section in ["business", "metric", "data", "feature", "model", "offline", "online", "serving", "monitor"]:
        assert section in text, f"missing: {section}"
    assert len(text.split()) >= 1000
