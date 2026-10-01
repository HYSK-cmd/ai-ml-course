"""MovieLens-1M loading and leakage-free splitting (pandas allowed here).

Files (latin-1, '::'-separated, no header):  ratings.dat UserID::MovieID::Rating::Timestamp
                                             movies.dat  MovieID::Title::Genres (pipe-separated)
"""
from __future__ import annotations

import pandas as pd


def load_ratings(root: str = "data/ml-1m") -> pd.DataFrame:
    """Columns: user_id, item_id, rating, timestamp (all int64)."""
    raise NotImplementedError


def load_movies(root: str = "data/ml-1m") -> pd.DataFrame:
    """Columns: item_id (int64), title (str), genres (list[str])."""
    raise NotImplementedError


def implicit(ratings: pd.DataFrame, min_rating: int = 4) -> pd.DataFrame:
    """Keep only interactions with rating >= min_rating (the 'positives')."""
    raise NotImplementedError


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2, min_train: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PER-USER temporal split: sort each user's interactions by (timestamp, item_id); the last
    ceil(test_frac * n) go to test, the rest to train. Users with fewer than min_train + 1
    interactions go entirely to train. Finally drop test rows whose item never appears in train
    (we can't retrieve an item we have no embedding for -- that's the item cold-start problem).
    Never shuffle: a random split leaks the future into training."""
    raise NotImplementedError


def encode_ids(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Map raw ids to contiguous ints built from TRAIN only: adds columns u and i to copies of both
    frames. Returns (train, test, user_index {raw: u}, item_index {raw: i}). Test rows with unknown
    users/items are dropped."""
    raise NotImplementedError
