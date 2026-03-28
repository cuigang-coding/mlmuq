from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


DATASET_SOURCES = {
    "ecs": {
        "name": "Education-Career-Success",
        "url": "https://www.kaggle.com/datasets/adilshamim8/education-and-career-success",
    },
    "jp": {
        "name": "Job-Placement",
        "url": "https://www.kaggle.com/datasets/mahad049/job-placement-dataset",
    },
    "ng": {
        "name": "Nigerian-Graduates",
        "url": "https://www.kaggle.com/code/obafemijoseph/nigerian-graduates",
    },
}


ACADEMIC_KW = (
    "gpa",
    "cgpa",
    "sat",
    "gre",
    "grade",
    "academic",
    "degree",
    "major",
    "class",
    "honor",
    "school",
    "university",
)
SKILL_KW = (
    "skill",
    "communication",
    "leadership",
    "cert",
    "competency",
    "language",
    "analytical",
    "technical",
    "problem",
)
EXPERIENCE_KW = (
    "intern",
    "experience",
    "work",
    "employment",
    "sector",
    "industry",
    "project",
    "volunteer",
    "part_time",
    "part-time",
)


@dataclass
class ModalityData:
    continuous: torch.Tensor
    categorical: torch.Tensor
    cardinalities: list[int]


@dataclass
class MetaDataset:
    modalities: dict[str, ModalityData]
    labels: torch.Tensor
    domains: torch.Tensor


@dataclass
class TaskEpisode:
    support_idx: torch.Tensor
    query_idx: torch.Tensor


class DataError(RuntimeError):
    pass


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        c = normalized.get(_normalize_col(cand))
        if c is not None:
            return c
    return None


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _bucket_salary_to_level(raw: pd.Series) -> pd.Series:
    # Output levels: 1..4 for employed samples (0 reserved for unemployed).
    if raw.dtype == object:
        text = raw.astype(str).str.lower()
        out = pd.Series(np.nan, index=raw.index, dtype=float)
        out[text.str.contains("<") | text.str.contains("low") | text.str.contains("30")] = 1
        out[text.str.contains("30-50") | text.str.contains("mid") | text.str.contains("2")] = 2
        out[text.str.contains("50-70") | text.str.contains("70-90") | text.str.contains("high") | text.str.contains("3")] = 3
        out[text.str.contains(">") | text.str.contains("premium") | text.str.contains("top") | text.str.contains("4") | text.str.contains("5")] = 4
        if out.notna().any():
            return out

    numeric = _to_numeric_series(raw)
    if numeric.notna().sum() < 5:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    # Robust quantiles for mixed datasets.
    q = numeric.rank(pct=True)
    out = pd.Series(index=raw.index, dtype=float)
    out[q <= 0.25] = 1
    out[(q > 0.25) & (q <= 0.50)] = 2
    out[(q > 0.50) & (q <= 0.75)] = 3
    out[q > 0.75] = 4
    return out


def _normalize_label_value(v: object) -> int | None:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        iv = int(v)
        if 0 <= iv <= 4:
            return iv
        if 1 <= iv <= 5:
            return iv - 1
        return None

    s = str(v).strip().lower()
    if "unemploy" in s or "not placed" in s or "fail" in s:
        return 0
    if "part-time" in s or "part time" in s or "self" in s or "non-standard" in s:
        return 1
    if "low" in s:
        return 1
    if "mid" in s or "medium" in s:
        return 2
    if "high" in s:
        return 3
    if "premium" in s or "top" in s:
        return 4
    return None


def _derive_ecs_label(df: pd.DataFrame) -> pd.Series:
    target_col = _find_col(
        df,
        [
            "career_success_score",
            "career success score",
            "career_success",
            "success_score",
            "target",
            "label",
            "outcome",
        ],
    )
    if target_col is None:
        raise DataError("ECS target column not found.")

    raw = df[target_col]
    mapped = raw.map(_normalize_label_value)
    if mapped.isna().all():
        numeric = _to_numeric_series(raw)
        mapped = numeric.round().astype("Int64") - 1
    mapped = mapped.astype("Int64")
    return mapped


def _derive_jp_label(df: pd.DataFrame) -> pd.Series:
    target_col = _find_col(df, ["target", "label", "placement_level", "career_success_score"])
    if target_col is not None:
        mapped = df[target_col].map(_normalize_label_value).astype("Int64")
        if mapped.notna().sum() > len(df) * 0.7:
            return mapped

    placement_col = _find_col(
        df,
        [
            "placement_status",
            "status",
            "placed",
            "is_placed",
            "job_offer",
            "employment_status",
        ],
    )
    salary_col = _find_col(df, ["salary", "salary_level", "salary_category", "ctc", "salary_usd"])

    if placement_col is None and salary_col is None:
        raise DataError("JP requires either placement or salary columns for label mapping.")

    placed = pd.Series(True, index=df.index)
    if placement_col is not None:
        ptxt = df[placement_col].astype(str).str.lower()
        placed = ~ptxt.str.contains("not|unemploy|no|0|false")

    salary_level = pd.Series(np.nan, index=df.index, dtype=float)
    if salary_col is not None:
        salary_level = _bucket_salary_to_level(df[salary_col])

    y = pd.Series(0, index=df.index, dtype=float)
    y.loc[placed] = salary_level.loc[placed].fillna(2)
    return y.astype("Int64")


def _derive_ng_label(df: pd.DataFrame) -> pd.Series:
    target_col = _find_col(df, ["target", "label", "employment_level", "outcome"])
    if target_col is not None:
        mapped = df[target_col].map(_normalize_label_value).astype("Int64")
        if mapped.notna().sum() > len(df) * 0.7:
            return mapped

    status_col = _find_col(
        df,
        [
            "employment_status",
            "status",
            "employment",
            "job_status",
        ],
    )
    salary_col = _find_col(df, ["salary_quintile", "salary_level", "salary", "income_quintile"])

    if status_col is None:
        raise DataError("NG employment status column not found.")

    status = df[status_col].astype(str).str.lower()
    salary_bucket = pd.Series(np.nan, index=df.index, dtype=float)
    if salary_col is not None:
        salary_bucket = _bucket_salary_to_level(df[salary_col])

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y[status.str.contains("unemploy")] = 0
    y[status.str.contains("part-time|part time|self")] = 1

    fulltime_mask = status.str.contains("full") | status.str.contains("employ")
    y.loc[fulltime_mask] = salary_bucket.loc[fulltime_mask].fillna(3)

    # Map employed salary buckets (1..4) to global levels (2..4).
    y = y.replace({1.0: 2.0, 2.0: 3.0, 3.0: 4.0, 4.0: 4.0})
    y[(status.str.contains("part-time|part time|self"))] = 1

    return y.astype("Int64")


def _assign_modality(col: str) -> str:
    c = _normalize_col(col)
    if any(k in c for k in ACADEMIC_KW):
        return "academic"
    if any(k in c for k in SKILL_KW):
        return "skills"
    if any(k in c for k in EXPERIENCE_KW):
        return "experience"
    return ""


def _build_modality_features(df: pd.DataFrame, labels_col: str, domain_col: str) -> dict[str, ModalityData]:
    feature_cols = [c for c in df.columns if c not in {labels_col, domain_col}]

    modality_cols: dict[str, list[str]] = {"academic": [], "skills": [], "experience": []}
    leftovers: list[str] = []

    for c in feature_cols:
        m = _assign_modality(c)
        if m:
            modality_cols[m].append(c)
        else:
            leftovers.append(c)

    # Keep balance when names are unknown.
    for c in leftovers:
        target = min(modality_cols, key=lambda m: len(modality_cols[m]))
        modality_cols[target].append(c)

    out: dict[str, ModalityData] = {}
    for m, cols in modality_cols.items():
        sub = df[cols].copy() if cols else pd.DataFrame(index=df.index)

        cont_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
        cat_cols = [c for c in sub.columns if c not in cont_cols]

        if cont_cols:
            cont = sub[cont_cols].astype(float)
            cont = cont.fillna(cont.mean(numeric_only=True))
            std = cont.std(axis=0).replace(0.0, 1.0)
            cont = (cont - cont.mean(axis=0)) / std
            cont_t = torch.tensor(cont.to_numpy(), dtype=torch.float32)
        else:
            cont_t = torch.empty((len(df), 0), dtype=torch.float32)

        cat_t_list: list[torch.Tensor] = []
        card: list[int] = []
        for c in cat_cols:
            values = sub[c].fillna("__MISSING__").astype(str)
            codes, uniques = pd.factorize(values)
            cat_t_list.append(torch.tensor(codes, dtype=torch.long).unsqueeze(1))
            card.append(max(2, len(uniques)))

        if cat_t_list:
            cat_t = torch.cat(cat_t_list, dim=1)
        else:
            cat_t = torch.empty((len(df), 0), dtype=torch.long)

        out[m] = ModalityData(continuous=cont_t, categorical=cat_t, cardinalities=card)

    return out


def load_and_preprocess(
    ecs_csv: Path,
    jp_csv: Path,
    ng_csv: Path,
) -> MetaDataset:
    ecs = pd.read_csv(ecs_csv)
    jp = pd.read_csv(jp_csv)
    ng = pd.read_csv(ng_csv)

    ecs["_y"] = _derive_ecs_label(ecs)
    jp["_y"] = _derive_jp_label(jp)
    ng["_y"] = _derive_ng_label(ng)

    ecs["_domain"] = 0
    jp["_domain"] = 1
    ng["_domain"] = 2

    common_cols = sorted(set(ecs.columns) | set(jp.columns) | set(ng.columns))
    ecs = ecs.reindex(columns=common_cols)
    jp = jp.reindex(columns=common_cols)
    ng = ng.reindex(columns=common_cols)

    full = pd.concat([ecs, jp, ng], ignore_index=True)
    full = full.dropna(subset=["_y"]).copy()
    full["_y"] = full["_y"].astype(int)

    # Keep only 0..4 mapped labels.
    full = full[(full["_y"] >= 0) & (full["_y"] <= 4)].reset_index(drop=True)
    if full.empty:
        raise DataError("No valid labels after preprocessing.")

    modalities = _build_modality_features(full, labels_col="_y", domain_col="_domain")

    labels = torch.tensor(full["_y"].to_numpy(), dtype=torch.long)
    domains = torch.tensor(full["_domain"].fillna(0).astype(int).to_numpy(), dtype=torch.long)
    return MetaDataset(modalities=modalities, labels=labels, domains=domains)


def sample_task_episodes(
    dataset: MetaDataset,
    num_tasks: int,
    support_per_class: int,
    query_per_class: int,
    num_classes: int = 5,
    seed: int | None = None,
) -> list[TaskEpisode]:
    rng = np.random.default_rng(seed)
    labels = dataset.labels.numpy()
    domains = dataset.domains.numpy()

    unique_domains = np.unique(domains)
    episodes: list[TaskEpisode] = []

    attempts = 0
    max_attempts = max(1000, num_tasks * 100)
    while len(episodes) < num_tasks and attempts < max_attempts:
        attempts += 1
        domain = int(rng.choice(unique_domains))
        domain_idx = np.where(domains == domain)[0]

        support_idx: list[int] = []
        query_idx: list[int] = []
        valid_classes = 0

        for c in range(num_classes):
            cls_idx = domain_idx[labels[domain_idx] == c]
            if len(cls_idx) == 0:
                continue
            valid_classes += 1
            need = support_per_class + query_per_class
            replace = len(cls_idx) < need
            sampled = rng.choice(cls_idx, size=need, replace=replace)
            support_idx.extend(sampled[:support_per_class].tolist())
            query_idx.extend(sampled[support_per_class:].tolist())

        if valid_classes < 2:
            continue
        episodes.append(
            TaskEpisode(
                support_idx=torch.tensor(support_idx, dtype=torch.long),
                query_idx=torch.tensor(query_idx, dtype=torch.long),
            )
        )

    if len(episodes) < num_tasks:
        raise DataError(f"Could only sample {len(episodes)} tasks; requested {num_tasks}.")

    return episodes


def batch_from_indices(
    dataset: MetaDataset,
    idx: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    x: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in dataset.modalities.items():
        x[name] = {
            "continuous": mod.continuous[idx].to(device),
            "categorical": mod.categorical[idx].to(device),
        }
    y = dataset.labels[idx].to(device)
    return x, y
