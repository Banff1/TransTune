# -*- coding: utf-8 -*-
"""
Extract PGTuner-compatible data_feature row from vector-db-benchmark data (datasets.json).

Supports ``type`` h5, jsonl, and tar (same layouts as benchmark readers).
Uses CPU + sklearn only (no cuML). For large N, pools rows before normalization to bound memory.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .registry import get_dataset, sklearn_metric, vdb_file_name
from .train_test_loaders import load_train_test

# Match PGTuner get_DS_feature k and get_DR_feature k
K_DS = 10
K_DR_QUERY = 100


def _normalize_rows(X: np.ndarray, distance: str) -> np.ndarray:
    if distance in ("cosine", "dot", "angular"):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return X / n
    return X


def _lid_mean_mle(
    X: np.ndarray, k: int, metric: str, sample_size: int, seed: int
) -> float:
    """Mean LID (MLE) over sampled points; simplified from ann-benchmark lid_calculator."""
    np.random.seed(seed)
    n = X.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(Xs)
    distances, _ = nn.kneighbors(Xs)
    knn = distances[:, 1:]
    lids = []
    for i in range(knn.shape[0]):
        d_k = knn[i, -1]
        if d_k <= 0:
            continue
        ratios = knn[i, :-1] / d_k
        ratios = ratios[ratios > 0]
        if len(ratios) == 0:
            continue
        log_sum = float(np.sum(np.log(ratios)))
        if log_sum == 0:
            continue
        lids.append(-1.0 / (log_sum / len(ratios)))
    if not lids:
        return float("nan")
    return float(np.nanmean(np.array(lids)))


def _ds_stats(vectors: np.ndarray, k: int, metric: str, max_points: int, seed: int) -> Tuple[float, float, float]:
    np.random.seed(seed)
    n = vectors.shape[0]
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        V = vectors[idx]
    else:
        V = vectors
    nn = NearestNeighbors(n_neighbors=min(k + 1, V.shape[0]), metric=metric)
    nn.fit(V)
    distances, _ = nn.kneighbors(V)
    if distances.shape[1] < 3:
        return 0.0, 0.0, 0.0
    sub = distances[:, 1:-1]
    whole_mean = float(np.mean(sub))
    if whole_mean <= 0:
        whole_mean = 1e-12
    sub = sub / whole_mean
    s = np.sum(sub, axis=1)
    return float(np.min(s)), float(np.max(s)), float(np.std(s))


def _dr_ratios(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    max_base: int,
    max_queries: int,
    seed: int,
) -> Tuple[float, float, float, float, float]:
    np.random.seed(seed)
    if base.shape[0] > max_base:
        bi = np.random.choice(base.shape[0], size=max_base, replace=False)
        B = base[bi]
    else:
        B = base
    if queries.shape[0] > max_queries:
        qi = np.random.choice(queries.shape[0], size=max_queries, replace=False)
        Q = queries[qi]
    else:
        Q = queries
    nn = NearestNeighbors(n_neighbors=min(1000, B.shape[0]), metric=metric)
    nn.fit(B)
    distances, _ = nn.kneighbors(Q)
    top = distances[:, :k]
    rest = distances[:, k:]
    m_top = np.mean(top, axis=1)
    m_rest = np.mean(rest, axis=1)
    ok = m_rest > 1e-12
    ratios = m_top[ok] / m_rest[ok]
    if ratios.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.min(ratios)),
        float(np.max(ratios)),
        float(np.mean(ratios)),
        float(np.std(ratios)),
        float(np.median(ratios)),
    )


def extract_row(
    dataset_name: str,
    *,
    data_path: Optional[Path] = None,
    h5_path: Optional[Path] = None,
    lid_sample: int = 8000,
    ds_max_points: int = 20000,
    dr_max_base: int = 80000,
    dr_max_queries: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Return one dict matching Data/data_feature.csv columns (no FileName prefix on keys except output).

    ``data_path`` overrides the default path from ``datasets.json`` (file for h5, directory for jsonl/tar).
    ``h5_path`` is deprecated; treated as an alias for ``data_path``.
    """
    override = data_path if data_path is not None else h5_path

    meta = get_dataset(dataset_name)
    distance = meta.get("distance", "cosine")
    metric = sklearn_metric(distance)
    train, test = load_train_test(dataset_name, data_path=override)

    n_tr = int(train.shape[0])
    n_te = int(test.shape[0])
    dim = int(meta.get("vector_size", int(train.shape[1])))

    need_tr = max(lid_sample, ds_max_points, dr_max_base)
    rng = np.random.RandomState(seed)
    if n_tr > need_tr:
        idxt = rng.choice(n_tr, size=need_tr, replace=False)
        train_pool = np.asarray(train[idxt], dtype=np.float32)
    else:
        train_pool = np.asarray(train, dtype=np.float32)
    train_n = _normalize_rows(train_pool, distance)

    if n_te > dr_max_queries:
        idxq = rng.choice(n_te, size=dr_max_queries, replace=False)
        test_pool = np.asarray(test[idxq], dtype=np.float32)
    else:
        test_pool = np.asarray(test, dtype=np.float32)
    test_n = _normalize_rows(test_pool, distance)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lid = _lid_mean_mle(train_n, k=20, metric=metric, sample_size=lid_sample, seed=seed)
        smin, smax, sstd = _ds_stats(train_n, K_DS, metric, ds_max_points, seed)
        qmin, qmax, qmean, qstd, qmed = _dr_ratios(
            train_n, test_n, K_DR_QUERY, metric, dr_max_base, dr_max_queries, seed
        )

    fn = vdb_file_name(dataset_name)
    return {
        "FileName": fn,
        "SIZE": float(n_tr),
        "DIM": float(dim),
        "LID": lid,
        "Sum_K_MinDist": smin,
        "Sum_K_MaxDist": smax,
        "Sum_K_StdDist": sstd,
        "q_SIZE": float(n_te),
        "q_K_MinRatio": qmin,
        "q_K_MaxRatio": qmax,
        "q_K_MeanRatio": qmean,
        "q_K_StdRatio": qstd,
    }


def upsert_data_feature_csv(row: Dict[str, Any], csv_path: Path) -> None:
    import pandas as pd

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "FileName",
        "SIZE",
        "DIM",
        "LID",
        "Sum_K_MinDist",
        "Sum_K_MaxDist",
        "Sum_K_StdDist",
        "q_SIZE",
        "q_K_MinRatio",
        "q_K_MaxRatio",
        "q_K_MeanRatio",
        "q_K_StdRatio",
    ]
    new = pd.DataFrame([{c: row[c] for c in cols}])
    if csv_path.is_file():
        old = pd.read_csv(csv_path)
        old = old[old["FileName"] != row["FileName"]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} ({len(out)} rows)")


def upsert_milvus_sizes(row_file_name: str, n: float, q: float, csv_path: Path) -> None:
    import pandas as pd

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{"FileName": row_file_name, "N": n, "q_queries": q}])
    if csv_path.is_file():
        old = pd.read_csv(csv_path)
        old = old[old["FileName"] != row_file_name]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} ({len(out)} rows)")
