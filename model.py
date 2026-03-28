from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import Predictive
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModalitySpec:
    continuous_dim: int
    categorical_cardinalities: list[int]


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        spec: ModalitySpec,
        embedding_dim: int,
        hidden_dims: tuple[int, int, int],
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spec = spec

        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in spec.categorical_cardinalities]
        )
        in_dim = spec.continuous_dim + len(spec.categorical_cardinalities) * embedding_dim

        self.empty = in_dim == 0
        if self.empty:
            self.empty_vec = nn.Parameter(torch.zeros(out_dim))
            self.mlp = nn.Identity()
            self.project = nn.Identity()
            return

        d1, d2, d3 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d1),
            nn.BatchNorm1d(d1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d2, d3),
            nn.BatchNorm1d(d3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.project = nn.Linear(d3, out_dim)

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if self.empty:
            return self.empty_vec.unsqueeze(0).expand(continuous.size(0), -1)

        parts = [continuous]
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, i]))
        x = torch.cat(parts, dim=-1)
        return self.project(self.mlp(x))


class UncertaintyWeightedCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.g_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.modality_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, m, _ = x.shape
        return x.view(b, m, self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, M, D], M = modality count (3 in MLMUQ)
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        g = torch.sigmoid(self._shape(self.g_proj(x)))

        scores = torch.einsum("bmhd,bnhd->bhmn", q, k) / (self.head_dim**0.5)

        sigma = self.modality_uncertainty(x).squeeze(-1) + 1e-6  # [B, M]
        sigma_pair = sigma.unsqueeze(1).unsqueeze(-1) * sigma.unsqueeze(1).unsqueeze(2)
        scores = scores / sigma_pair

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhmn,bnhd->bmhd", attn, v) * g
        fused = ctx.reshape(x.size(0), x.size(1), self.hidden_dim)

        z = self.out(fused.mean(dim=1))
        return z, attn, sigma


class MLMUQBackbone(nn.Module):
    def __init__(
        self,
        modality_specs: dict[str, ModalitySpec],
        embedding_dim: int,
        hidden_dims: tuple[int, int, int],
        hidden_dim: int,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.modality_order = ["academic", "skills", "experience"]

        self.encoders = nn.ModuleDict(
            {
                m: ModalityEncoder(
                    spec=modality_specs[m],
                    embedding_dim=embedding_dim,
                    hidden_dims=hidden_dims,
                    out_dim=hidden_dim,
                    dropout=dropout,
                )
                for m in self.modality_order
            }
        )
        self.fusion = UncertaintyWeightedCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(self, x: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reps = []
        for m in self.modality_order:
            reps.append(self.encoders[m](x[m]["continuous"], x[m]["categorical"]))
        stacked = torch.stack(reps, dim=1)
        z, attn, sigma = self.fusion(stacked)
        return z, attn, sigma


class MLMUQModel(nn.Module):
    def __init__(
        self,
        backbone: MLMUQBackbone,
        hidden_dim: int,
        num_classes: int,
        kl_weight: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.kl_weight = kl_weight

        self.register_buffer("prior_loc_w", torch.zeros(num_classes, hidden_dim))
        self.register_buffer("prior_scale_w", torch.ones(num_classes, hidden_dim))
        self.register_buffer("prior_loc_b", torch.zeros(num_classes))
        self.register_buffer("prior_scale_b", torch.ones(num_classes))

    def encode(self, x: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        z, _, _ = self.backbone(x)
        return z

    def probabilistic_model(
        self,
        x: dict[str, dict[str, torch.Tensor]],
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encode(x)

        with poutine.scale(scale=self.kl_weight):
            w = pyro.sample("w_cls", dist.Normal(self.prior_loc_w, self.prior_scale_w).to_event(2))
            b = pyro.sample("b_cls", dist.Normal(self.prior_loc_b, self.prior_scale_b).to_event(1))

        logits = torch.matmul(z, w.transpose(0, 1)) + b
        pyro.deterministic("logits", logits)

        with pyro.plate("data", size=z.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits

    def posterior_mean_logits(
        self,
        x: dict[str, dict[str, torch.Tensor]],
        guide: Any | None,
    ) -> torch.Tensor:
        z = self.encode(x)
        if guide is None:
            w, b = self.prior_loc_w, self.prior_loc_b
        else:
            try:
                med = guide.median()
                w = med.get("w_cls", self.prior_loc_w)
                b = med.get("b_cls", self.prior_loc_b)
            except Exception:
                w, b = self.prior_loc_w, self.prior_loc_b
        return torch.matmul(z, w.transpose(0, 1)) + b

    @torch.no_grad()
    def uncertainty_decomposition(
        self,
        x: dict[str, dict[str, torch.Tensor]],
        guide: Any,
        num_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        predictive = Predictive(
            model=self.probabilistic_model,
            guide=guide,
            num_samples=num_samples,
            return_sites=("logits",),
        )
        sampled = predictive(x)
        logits = sampled["logits"]  # [S, B, C]

        probs = torch.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        predictive_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)
        expected_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean(dim=0)

        epistemic = (predictive_entropy - expected_entropy).clamp_min(0.0)
        aleatoric = expected_entropy.clamp_min(0.0)

        return {
            "mean_probs": mean_probs,
            "predictive_entropy": predictive_entropy,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }

    @torch.no_grad()
    def update_data_dependent_prior(self, guide: Any, prior_reg: float = 0.01) -> None:
        """
        Update p_meta from posterior quantiles every fixed outer iterations.

        prior_reg is a small shrinkage toward N(0, I) to avoid overfitting.
        """
        try:
            q = guide.quantiles([0.16, 0.5, 0.84])
            w_q = q["w_cls"]
            b_q = q["b_cls"]
        except Exception:
            return

        w_loc = w_q[1]
        b_loc = b_q[1]
        w_scale = (w_q[2] - w_q[0]).abs() / 2.0
        b_scale = (b_q[2] - b_q[0]).abs() / 2.0

        # Eq. prior regularization term: lambda_reg ||mu_meta||^2
        self.prior_loc_w.copy_((1.0 - prior_reg) * w_loc)
        self.prior_loc_b.copy_((1.0 - prior_reg) * b_loc)

        self.prior_scale_w.copy_(((1.0 - prior_reg) * w_scale + prior_reg).clamp_min(1e-3))
        self.prior_scale_b.copy_(((1.0 - prior_reg) * b_scale + prior_reg).clamp_min(1e-3))
