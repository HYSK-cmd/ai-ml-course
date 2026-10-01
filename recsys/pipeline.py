"""Glue: build the full two-stage recommender from raw data.

    python -m recsys.pipeline        # trains everything and prints the offline evaluation table
"""
from __future__ import annotations

from dataclasses import dataclass, field

from recsys.two_tower import TwoTowerConfig


@dataclass
class PipelineConfig:
    data_root: str = "data/ml-1m"
    min_rating: int = 4
    test_frac: float = 0.2
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    n_lists: int = 64
    n_probe: int = 8
    n_candidates: int = 200     # retrieved per user before re-ranking
    use_ranker: bool = False    # stretch


class Recommender:
    """Serving-time object. recommend(raw_user_id, k) -> list of raw movie ids, excluding movies the
    user interacted with in train. Unknown users (cold start) get the popularity fallback."""

    def recommend(self, raw_user_id: int, k: int = 10) -> tuple[list[int], bool]:
        """-> (raw item ids, used_fallback)."""
        raise NotImplementedError


def build(cfg: PipelineConfig) -> tuple[Recommender, dict]:
    """Load, split, train two-tower (+ ranker), build the IVF index. Returns the recommender and a
    context dict with at least: "train", "test" (encoded frames), "user_index", "item_index",
    "user_vecs", "item_vecs", "index" (the IVFIndex)."""
    raise NotImplementedError


def evaluate(rec: Recommender, ctx: dict, k: int = 10, recall_k: int = 50) -> dict:
    """Offline eval on the test split: {"recall@50", "ndcg@10", "hit@10", "coverage@10"} for the full
    recommender and for the popularity baseline ({"pop_recall@50", ...})."""
    raise NotImplementedError


if __name__ == "__main__":
    recommender, context = build(PipelineConfig())
    print(evaluate(recommender, context))
