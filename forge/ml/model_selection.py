from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int | None = None):
    """Shuffle, then split. Returns X_train, X_test, y_train, y_test. n_test = ceil(test_size * n)."""
    raise NotImplementedError


class KFold:
    """Same fold sizes as sklearn: the first n % k folds have one extra sample.
    shuffle=False -> contiguous folds in order (must equal sklearn's KFold exactly)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        raise NotImplementedError

    def split(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (train_indices, test_indices)."""
        raise NotImplementedError


def cross_val_score(make_model: Callable[[], object], X: np.ndarray, y: np.ndarray, cv: int | KFold = 5,
                    scoring: Callable[[np.ndarray, np.ndarray], float] | None = None) -> np.ndarray:
    """Fit a FRESH model (make_model()) on each training fold, score predictions on the held-out fold.
    cv int -> KFold(cv, shuffle=True, random_state=0). scoring default: accuracy."""
    raise NotImplementedError
