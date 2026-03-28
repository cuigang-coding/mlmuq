from __future__ import annotations

import copy
from dataclasses import dataclass

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import torch
import torch.nn.functional as F

from .config import MLMUQConfig
from .data import MetaDataset, batch_from_indices, sample_task_episodes
from .model import MLMUQModel


@dataclass
class TrainState:
    iteration: int = 0
    avg_support_loss: float = 0.0
    avg_query_loss: float = 0.0
    avg_svi_loss: float = 0.0


class MetaTrainer:
    def __init__(
        self,
        model: MLMUQModel,
        cfg: MLMUQConfig,
        dataset: MetaDataset,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.dataset = dataset
        self.device = device

        pyro.clear_param_store()
        self.guide = AutoDiagonalNormal(self.model.probabilistic_model)
        self.svi = SVI(
            self.model.probabilistic_model,
            self.guide,
            pyro.optim.Adam({"lr": cfg.outer_lr}),
            loss=Trace_ELBO(),
        )

        # Fallback deterministic head until guide posterior stabilizes.
        self.fallback_w = torch.randn(cfg.num_classes, cfg.hidden_dim, device=device) * 0.02
        self.fallback_b = torch.zeros(cfg.num_classes, device=device)

    def _posterior_mean_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            med = self.guide.median()
            w = med.get("w_cls")
            b = med.get("b_cls")
            if w is None or b is None:
                raise RuntimeError("posterior sites unavailable")
            return w.detach(), b.detach()
        except Exception:
            return self.fallback_w, self.fallback_b

    def _adapt_backbone(
        self,
        x_support: dict[str, dict[str, torch.Tensor]],
        y_support: torch.Tensor,
    ) -> tuple[torch.nn.Module, float]:
        w, b = self._posterior_mean_head()

        task_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        task_optimizer = torch.optim.SGD(task_backbone.parameters(), lr=self.cfg.inner_lr)

        support_loss_val = 0.0
        for _ in range(self.cfg.inner_steps):
            z, _, _ = task_backbone(x_support)
            logits = z @ w.transpose(0, 1) + b
            loss = F.cross_entropy(logits, y_support)

            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()

            support_loss_val = float(loss.item())

        return task_backbone, support_loss_val

    @torch.no_grad()
    def _reptile_outer_update(self, adapted_states: list[dict[str, torch.Tensor]]) -> None:
        if not adapted_states:
            return

        current = self.model.backbone.state_dict()
        mean_target: dict[str, torch.Tensor] = {}
        for k in current.keys():
            stacked = torch.stack([s[k] for s in adapted_states], dim=0)
            mean_target[k] = stacked.mean(dim=0)

        updated = {}
        for k, v in current.items():
            updated[k] = v + self.cfg.outer_lr * (mean_target[k] - v)

        self.model.backbone.load_state_dict(updated)

    def train(self) -> TrainState:
        state = TrainState()

        for it in range(1, self.cfg.outer_iterations + 1):
            episodes = sample_task_episodes(
                dataset=self.dataset,
                num_tasks=self.cfg.meta_batch_size,
                support_per_class=self.cfg.support_per_class,
                query_per_class=self.cfg.query_per_class,
                num_classes=self.cfg.num_classes,
                seed=self.cfg.seed + it,
            )

            adapted_states: list[dict[str, torch.Tensor]] = []
            support_losses: list[float] = []
            query_losses: list[float] = []
            svi_losses: list[float] = []

            for ep in episodes:
                x_spt, y_spt = batch_from_indices(self.dataset, ep.support_idx, self.device)
                x_qry, y_qry = batch_from_indices(self.dataset, ep.query_idx, self.device)

                # Inner-loop adaptation on support set (MAML-style fast adaptation).
                task_backbone, support_loss = self._adapt_backbone(x_spt, y_spt)
                adapted_states.append({k: v.detach().clone() for k, v in task_backbone.state_dict().items()})
                support_losses.append(support_loss)

                # Query loss with adapted backbone (for reporting).
                w, b = self._posterior_mean_head()
                z_q, _, _ = task_backbone(x_qry)
                q_logits = z_q @ w.transpose(0, 1) + b
                q_loss = F.cross_entropy(q_logits, y_qry)
                query_losses.append(float(q_loss.item()))

                # Variational update of Bayesian head on query sets.
                for _ in range(self.cfg.svi_steps_per_iter):
                    svi_loss = self.svi.step(x_qry, y_qry)
                    svi_losses.append(float(svi_loss) / max(1, len(y_qry)))

            self._reptile_outer_update(adapted_states)

            if it % self.cfg.prior_update_every == 0:
                self.model.update_data_dependent_prior(self.guide, prior_reg=self.cfg.prior_reg)

            state = TrainState(
                iteration=it,
                avg_support_loss=float(sum(support_losses) / max(1, len(support_losses))),
                avg_query_loss=float(sum(query_losses) / max(1, len(query_losses))),
                avg_svi_loss=float(sum(svi_losses) / max(1, len(svi_losses))),
            )

            if it % 50 == 0 or it == 1:
                print(
                    f"[iter {it:04d}] support={state.avg_support_loss:.4f} "
                    f"query={state.avg_query_loss:.4f} svi={state.avg_svi_loss:.4f}"
                )

        return state

    @torch.no_grad()
    def evaluate(self, num_tasks: int = 30) -> dict[str, float]:
        episodes = sample_task_episodes(
            dataset=self.dataset,
            num_tasks=num_tasks,
            support_per_class=self.cfg.support_per_class,
            query_per_class=self.cfg.query_per_class,
            num_classes=self.cfg.num_classes,
            seed=self.cfg.seed + 9999,
        )

        total = 0
        correct = 0
        epi_vals = []
        ale_vals = []

        for ep in episodes:
            x_qry, y_qry = batch_from_indices(self.dataset, ep.query_idx, self.device)
            unc = self.model.uncertainty_decomposition(
                x_qry,
                guide=self.guide,
                num_samples=self.cfg.mc_samples,
            )
            pred = unc["mean_probs"].argmax(dim=-1)

            total += int(y_qry.numel())
            correct += int((pred == y_qry).sum().item())

            epi_vals.append(float(unc["epistemic"].mean().item()))
            ale_vals.append(float(unc["aleatoric"].mean().item()))

        return {
            "accuracy": correct / max(1, total),
            "epistemic_mean": sum(epi_vals) / max(1, len(epi_vals)),
            "aleatoric_mean": sum(ale_vals) / max(1, len(ale_vals)),
        }
