from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """The baseline every recommender must beat: most-interacted items in train.
    Ties broken by smaller item id."""

    def fit(self, train: pd.DataFrame, item_col: str = "i") -> PopularityRecommender:
        raise NotImplementedError

    def recommend(self, k: int, exclude: set | None = None) -> list[int]:
        raise NotImplementedError
