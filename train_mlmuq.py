from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pyro
import torch

from .config import MLMUQConfig
from .data import load_and_preprocess
from .meta_trainer import MetaTrainer
from .model import MLMUQBackbone, MLMUQModel, ModalitySpec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLMUQ training (PyTorch + Pyro)")

    p.add_argument("--ecs-csv", type=Path, required=True, help="Path to ECS CSV")
    p.add_argument("--jp-csv", type=Path, required=True, help="Path to JP CSV")
    p.add_argument("--ng-csv", type=Path, required=True, help="Path to NG CSV")

    p.add_argument("--outer-iters", type=int, default=2000)
    p.add_argument("--meta-batch-size", type=int, default=8)
    p.add_argument("--support-per-class", type=int, default=15)
    p.add_argument("--query-per-class", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp-dir", type=Path, default=Path("runs/mlmuq"))
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = MLMUQConfig(
        seed=args.seed,
        outer_iterations=args.outer_iters,
        meta_batch_size=args.meta_batch_size,
        support_per_class=args.support_per_class,
        query_per_class=args.query_per_class,
        experiment_dir=args.exp_dir,
    )

    set_seed(cfg.seed)

    print("Loading and preprocessing datasets...")
    dataset = load_and_preprocess(args.ecs_csv, args.jp_csv, args.ng_csv)

    modality_specs = {
        name: ModalitySpec(
            continuous_dim=mod.continuous.shape[1],
            categorical_cardinalities=mod.cardinalities,
        )
        for name, mod in dataset.modalities.items()
    }

    backbone = MLMUQBackbone(
        modality_specs=modality_specs,
        embedding_dim=cfg.embedding_dim,
        hidden_dims=cfg.encoder_dims,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        num_heads=cfg.attention_heads,
    )
    model = MLMUQModel(
        backbone=backbone,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        kl_weight=cfg.kl_weight,
    )

    device = torch.device(args.device)
    trainer = MetaTrainer(model=model, cfg=cfg, dataset=dataset, device=device)

    print("Start meta-training...")
    final_state = trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate(num_tasks=30)

    exp_dir = cfg.experiment_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = exp_dir / "mlmuq_checkpoint.pt"
    param_store = {k: v.detach().cpu() for k, v in pyro.get_param_store().items()}
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "prior_loc_w": model.prior_loc_w.detach().cpu(),
            "prior_scale_w": model.prior_scale_w.detach().cpu(),
            "prior_loc_b": model.prior_loc_b.detach().cpu(),
            "prior_scale_b": model.prior_scale_b.detach().cpu(),
            "pyro_param_store": param_store,
        },
        ckpt_path,
    )

    summary = {
        "final_iteration": final_state.iteration,
        "avg_support_loss": final_state.avg_support_loss,
        "avg_query_loss": final_state.avg_query_loss,
        "avg_svi_loss": final_state.avg_svi_loss,
        "eval": metrics,
        "device": str(device),
        "outer_iterations": cfg.outer_iterations,
        "inner_steps": cfg.inner_steps,
        "inner_lr": cfg.inner_lr,
        "outer_lr": cfg.outer_lr,
        "kl_weight": cfg.kl_weight,
    }

    with (exp_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
