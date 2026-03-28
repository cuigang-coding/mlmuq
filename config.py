from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLMUQConfig:
    """Configuration aligned with the manuscript implementation section."""

    seed: int = 42
    num_classes: int = 5

    # Paper-aligned architecture.
    hidden_dim: int = 512
    encoder_dims: tuple[int, int, int] = (512, 256, 128)
    embedding_dim: int = 32
    dropout: float = 0.30
    attention_heads: int = 4

    # Meta-learning + Bayesian optimization (paper defaults).
    inner_lr: float = 0.01
    inner_steps: int = 5
    outer_lr: float = 1e-3
    meta_batch_size: int = 8
    outer_iterations: int = 2000
    svi_steps_per_iter: int = 1
    kl_weight: float = 0.05

    # Data-dependent prior update.
    prior_update_every: int = 10
    prior_reg: float = 0.01

    # Episodic sampling.
    support_per_class: int = 15
    query_per_class: int = 20

    # Posterior MC samples for uncertainty decomposition.
    mc_samples: int = 30

    # I/O
    experiment_dir: Path = field(default_factory=lambda: Path("runs/mlmuq"))
