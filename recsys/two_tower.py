"""Two-tower retrieval model (YouTube 2019 / Google 'sampling-bias-corrected' paper), PyTorch.

user tower: Embedding(n_users, dim) -> MLP -> L2-normalize
item tower: Embedding(n_items, dim) + Linear(genre multi-hot) -> MLP -> L2-normalize
score(u, i) = <user_vec, item_vec> / temperature

Training: batches of (u, i) positive pairs. IN-BATCH NEGATIVES: logits = U @ I^T / temperature
(B x B), label for row r is column r, loss = cross-entropy.
CORE: logQ correction (cfg.logq_correction, on by default). Popular items show up as in-batch
negatives far more often than rare ones, so the uncorrected model learns to push popular items
DOWN -- on MovieLens it then barely ties the popularity baseline. Subtract log(q_j) (item j's
frequency in the training stream) from every logit column during training only. Train once with
it off and once on and compare; the week-8 lesson explains the math.
Also mask false negatives: when the same item appears twice in a batch, the off-diagonal copy is
not a negative (set that logit to -inf).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class TwoTowerConfig:
    dim: int = 64
    hidden: int = 128
    temperature: float = 0.05
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 3e-3
    weight_decay: float = 1e-5
    logq_correction: bool = True
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, item_genres: torch.Tensor, cfg: TwoTowerConfig):
        """item_genres: float (n_items, n_genres) multi-hot, registered as a buffer."""
        super().__init__()
        raise NotImplementedError

    def user_vecs(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def item_vecs(self, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """In-batch logits (B, B)."""
        raise NotImplementedError


def genre_matrix(movies: pd.DataFrame, item_index: dict) -> tuple[np.ndarray, list[str]]:
    """(n_items, n_genres) multi-hot float32 in item_index order, plus the sorted genre names."""
    raise NotImplementedError


def train_two_tower(train: pd.DataFrame, n_users: int, n_items: int, item_genres: np.ndarray,
                    cfg: TwoTowerConfig) -> TwoTower:
    """Train on (u, i) rows of ``train``. Returns the model in eval mode."""
    raise NotImplementedError


@torch.no_grad()
def export_embeddings(model: TwoTower) -> tuple[np.ndarray, np.ndarray]:
    """(user_vecs (n_users, dim), item_vecs (n_items, dim)) as float32 numpy arrays."""
    raise NotImplementedError
