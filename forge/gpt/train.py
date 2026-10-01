"""Training loop: AdamW, warmup + cosine LR, gradient clipping, bf16 autocast on GPU, checkpointing.

    python -m forge.gpt.train                      # default config, trains on data/shakespeare.txt

Week 6 adds MLflow tracking (cfg.mlflow=True): log every TrainConfig field as a param, train/val loss
as metrics (with step), and the checkpoint file as an artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from forge.gpt.model import GPT


@dataclass
class TrainConfig:
    data_path: str = "data/shakespeare.txt"
    tokenizer: str = "char"            # "char" or "bpe"
    bpe_vocab_size: int = 512
    val_frac: float = 0.1              # last 10% of the token stream is validation
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    batch_size: int = 64
    max_steps: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1          # only on >= 2-D params (matrices/embeddings), not biases/LayerNorm
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    amp: bool = True                   # bf16 autocast when on cuda
    seed: int = 1337
    out_dir: str = "checkpoints/gpt"
    mlflow: bool = False               # Week 6
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "forge-gpt"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """data: 1-D int64 token stream. Sample batch_size random windows; y is x shifted left by one.
    Every window must fit: start in [0, len(data) - block_size - 1]."""
    raise NotImplementedError


def lr_at(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """step < warmup: linear max_lr * (step + 1) / warmup_steps.
    step >= max_steps: min_lr.  Otherwise cosine from max_lr down to min_lr over [warmup, max_steps]."""
    raise NotImplementedError


def save_checkpoint(path: str, model: GPT, tokenizer, step: int | None = None) -> None:
    """One file holding model config, state_dict and the tokenizer (so load_checkpoint needs nothing else)."""
    raise NotImplementedError


def load_checkpoint(path: str, device: str = "cpu"):
    """-> (model in eval mode on device, tokenizer)."""
    raise NotImplementedError


def train(cfg: TrainConfig) -> dict:
    """Full run. Returns {"train_loss": float, "val_loss": float, "steps": int, "checkpoint": str}
    where the losses are the final estimates (mean over eval_iters batches, model in eval mode)."""
    raise NotImplementedError


if __name__ == "__main__":
    print(train(TrainConfig()))
