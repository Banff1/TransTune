"""
Task profiles and similarity from aggregated Prometheus metrics.

OtterTune paper: workload metrics over an observation period → factor analysis → similarity.
Here we use per-metric aggregates (mean/std) and optional PCA for dimensionality reduction,
then L2 / cosine distance between tasks (same canonical workload for each task).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

DistanceMetric = Literal["euclidean", "cosine"]


@dataclass
class TaskProfile:
    """One tuning task / workload fingerprint after running the same probe workload."""

    task_id: str
    metric_names: List[str]
    mean_vector: np.ndarray
    std_vector: np.ndarray
    num_samples: int
    raw_sample_count_per_metric: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "metric_names": self.metric_names,
            "mean_vector": self.mean_vector.tolist(),
            "std_vector": self.std_vector.tolist(),
            "num_samples": self.num_samples,
            "raw_sample_count_per_metric": self.raw_sample_count_per_metric,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskProfile":
        return cls(
            task_id=d["task_id"],
            metric_names=list(d["metric_names"]),
            mean_vector=np.asarray(d["mean_vector"], dtype=np.float64),
            std_vector=np.asarray(d["std_vector"], dtype=np.float64),
            num_samples=int(d["num_samples"]),
            raw_sample_count_per_metric=dict(d.get("raw_sample_count_per_metric") or {}),
            meta=dict(d.get("meta") or {}),
        )


def aggregate_observation(
    history: Sequence[Dict[str, Any]],
    exclude_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build mean/std vectors over observation rows (DataFrame-like list of dicts).

    Keys like timestamp_epoch, sample_index are excluded from metrics.

    ``raw_sample_count_per_metric`` (the returned counts dict): for each metric column, how many
    sample rows contained a numeric value. If upstream uses fill_missing=True, every row has every
    metric key, so **all counts equal the number of observation rows** — that does not mean each
    metric had the same *Prometheus* data; see prometheus_query_success_samples_per_metric in meta.
    """
    exclude = set(exclude_keys or ("timestamp_epoch", "sample_index", "timestamp"))
    if not history:
        return [], np.array([], dtype=np.float64), np.array([], dtype=np.float64), {}

    # Union of all metric keys across samples
    keys: List[str] = []
    seen = set()
    for row in history:
        for k in row:
            if k in exclude or k in seen:
                continue
            v = row[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                seen.add(k)
                keys.append(k)
    keys.sort()

    n = len(history)
    m = len(keys)
    mat = np.full((n, m), np.nan, dtype=np.float64)
    counts = {k: 0 for k in keys}

    for i, row in enumerate(history):
        for j, k in enumerate(keys):
            if k in row and row[k] is not None:
                try:
                    mat[i, j] = float(row[k])
                    counts[k] += 1
                except (TypeError, ValueError):
                    pass

    means = np.nanmean(mat, axis=0)
    stds = np.nanstd(mat, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=0.0)
    return keys, means, stds, counts


def _standardize_profiles(
    profiles: Sequence[TaskProfile],
    use_std: bool = True,
    zscore_across_tasks: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Stack mean (and optional std) vectors; optionally z-score each column across tasks."""
    if not profiles:
        return np.zeros((0, 0)), []
    names = profiles[0].metric_names
    for p in profiles:
        if p.metric_names != names:
            raise ValueError("All profiles must share the same metric_names ordering")
    X = np.stack([p.mean_vector for p in profiles], axis=0)
    S = np.stack([p.std_vector for p in profiles], axis=0) if use_std else None

    if zscore_across_tasks:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Zm = (X - mu) / sigma
        if use_std and S is not None:
            mu_s = S.mean(axis=0)
            sig_s = S.std(axis=0)
            sig_s = np.where(sig_s < 1e-12, 1.0, sig_s)
            Zs = (S - mu_s) / sig_s
            Z = np.hstack([Zm, Zs])
        else:
            Z = Zm
    else:
        if use_std and S is not None:
            Z = np.hstack([X, S])
        else:
            Z = X
    return Z, names


def optional_pca(X: np.ndarray, n_components: Optional[int]) -> np.ndarray:
    if n_components is None or n_components <= 0 or X.size == 0:
        return X
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        LOG.warning("sklearn not installed; skipping PCA")
        return X
    k = min(n_components, X.shape[0], X.shape[1])
    if k < 1:
        return X
    pca = PCA(n_components=k)
    return pca.fit_transform(X)


def optional_factor_analysis(X: np.ndarray, n_components: Optional[int]) -> np.ndarray:
    """OtterTune-style linear latent factors (paper Sec. 4 / factor analysis)."""
    if n_components is None or n_components <= 0 or X.size == 0:
        return X
    try:
        from sklearn.decomposition import FactorAnalysis
    except ImportError:
        LOG.warning("sklearn not installed; skipping FactorAnalysis")
        return X
    k = min(n_components, X.shape[0], X.shape[1])
    if k < 1:
        return X
    fa = FactorAnalysis(n_components=k, random_state=0)
    return fa.fit_transform(X)


def pairwise_distance_matrix(
    profiles: Sequence[TaskProfile],
    *,
    metric: DistanceMetric = "euclidean",
    zscore_across_tasks: bool = True,
    include_std_in_features: bool = True,
    pca_components: Optional[int] = None,
    fa_components: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Pairwise distances between tasks (lower = more similar workloads under same probe).

    :return: (D, task_ids) where D[i,j] is distance between task i and j
    """
    ids = [p.task_id for p in profiles]
    X, _ = _standardize_profiles(
        profiles,
        use_std=include_std_in_features,
        zscore_across_tasks=zscore_across_tasks,
    )
    use_fa = fa_components is not None and fa_components > 0
    use_pca = pca_components is not None and pca_components > 0
    if X.size and (use_fa or use_pca):
        if use_fa:
            X = optional_factor_analysis(X, fa_components)
        else:
            X = optional_pca(X, pca_components)

    n = len(profiles)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            a, b = X[i], X[j]
            if metric == "euclidean":
                d = float(np.linalg.norm(a - b))
            elif metric == "cosine":
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-12 or nb < 1e-12:
                    d = 1.0
                else:
                    sim = float(np.dot(a, b) / (na * nb))
                    sim = max(-1.0, min(1.0, sim))
                    d = 1.0 - sim
            else:
                raise ValueError(metric)
            D[i, j] = D[j, i] = d
    return D, ids


def save_profile(profile: TaskProfile, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_profile(path: Path) -> TaskProfile:
    with open(path, "r", encoding="utf-8") as f:
        return TaskProfile.from_dict(json.load(f))


def load_profiles_from_dir(directory: Path) -> List[TaskProfile]:
    directory = Path(directory)
    out: List[TaskProfile] = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(load_profile(p))
        except Exception as e:
            LOG.warning("Skip %s: %s", p, e)
    return out
