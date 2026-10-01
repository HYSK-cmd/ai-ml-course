from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class DataLoader:
    """Mini-batches over in-memory arrays. Yields (X_batch, y_batch) as numpy arrays.
    shuffle=True -> a NEW permutation every epoch (every __iter__), reproducible from ``seed``.
    drop_last=True -> skip the final short batch. ``len(loader)`` = number of batches per epoch."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = False,
                 drop_last: bool = False, seed: int | None = None):
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
