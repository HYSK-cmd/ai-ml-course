"""Offline ranking metrics. ``recs``: {user: ranked list of items}; ``truth``: {user: set of relevant
items}. Every metric averages over users that have a NON-EMPTY truth set (users missing from recs
count as empty recommendation lists)."""
from __future__ import annotations


def recall_at_k(recs: dict, truth: dict, k: int) -> float:
    """|top-k ∩ truth| / |truth|"""
    raise NotImplementedError


def hit_rate_at_k(recs: dict, truth: dict, k: int) -> float:
    """1 if top-k contains any relevant item else 0."""
    raise NotImplementedError


def ndcg_at_k(recs: dict, truth: dict, k: int) -> float:
    """Binary relevance. DCG = sum over hits at 1-based rank r of 1 / log2(r + 1);
    IDCG = the DCG of min(k, |truth|) hits at ranks 1, 2, ..."""
    raise NotImplementedError


def catalog_coverage(recs: dict, n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top-k."""
    raise NotImplementedError
