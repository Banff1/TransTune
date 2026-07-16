#!/usr/bin/env python3
"""Generate a sampled dataset while keeping query vectors unchanged."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_ADAPT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = NEW_ADAPT_DIR.parent.parent.parent
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "vector-db-benchmark-master" / "datasets"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASETS_ROOT / "new_adapt"

sys.path.insert(0, str(NEW_ADAPT_DIR))
from generate_drift_dataset import resolve_source


def iter_jsonl(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def sanitize_dataset_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    name = name.strip("-._")
    return name or "custom-dataset"


def infer_top_k(source_dir: Path, fallback: int) -> int:
    neighbours_path = source_dir / "neighbours.jsonl"
    if not neighbours_path.exists():
        return fallback
    with neighbours_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            values = json.loads(line)
            if isinstance(values, list) and values:
                return len(values)
    return fallback


def count_train_vectors_jsonl(vectors_path: Path) -> int:
    count = 0
    with vectors_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                count += 1
    return count


def load_queries_jsonl(queries_path: Path) -> np.ndarray:
    queries = [np.asarray(item, dtype=np.float32) for item in iter_jsonl(queries_path)]
    if not queries:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(queries)


def load_train_matrix(source) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for _, vector in source.iter_train_vectors():
        vectors.append(np.asarray(vector, dtype=np.float32))
    if not vectors:
        return np.empty((0, source.vector_size), dtype=np.float32)
    return np.vstack(vectors)


def resolve_kmeans_clusters(requested_clusters: int, total_vectors: int) -> int:
    if total_vectors <= 0:
        return 1
    if requested_clusters > 0:
        return min(requested_clusters, total_vectors)
    heuristic = int(np.sqrt(total_vectors))
    return max(1, min(100, heuristic))


def compute_stratified_cluster_quotas(cluster_sizes: np.ndarray, sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros_like(cluster_sizes, dtype=np.int64)
    if sample_count >= int(cluster_sizes.sum()):
        return cluster_sizes.astype(np.int64, copy=True)

    expected = cluster_sizes.astype(np.float64) * (sample_count / float(cluster_sizes.sum()))
    quotas = np.floor(expected).astype(np.int64)
    quotas = np.minimum(quotas, cluster_sizes)
    remaining = sample_count - int(quotas.sum())
    if remaining <= 0:
        return quotas

    fractional = expected - quotas
    order = np.argsort(-fractional)
    for cluster_id in order:
        if remaining <= 0:
            break
        if quotas[cluster_id] < cluster_sizes[cluster_id]:
            quotas[cluster_id] += 1
            remaining -= 1

    while remaining > 0:
        capacities = cluster_sizes - quotas
        eligible = np.flatnonzero(capacities > 0)
        if eligible.size == 0:
            break
        pick = int(eligible[np.argmax(capacities[eligible])])
        quotas[pick] += 1
        remaining -= 1
    return quotas


def sample_ids_by_kmeans_stratified(
    train_matrix: np.ndarray, sample_count: int, seed: int, n_clusters: int, batch_size: int
) -> tuple[list[int], np.ndarray, int]:
    total_vectors = int(train_matrix.shape[0])
    if sample_count >= total_vectors:
        labels = np.zeros(total_vectors, dtype=np.int32)
        return list(range(total_vectors)), labels, 1

    effective_clusters = max(1, min(n_clusters, total_vectors))
    if effective_clusters == 1:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(total_vectors, size=sample_count, replace=False).tolist())
        labels = np.zeros(total_vectors, dtype=np.int32)
        return selected, labels, 1

    kmeans = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=seed,
        batch_size=max(256, batch_size),
        n_init="auto",
    )
    labels = kmeans.fit_predict(train_matrix).astype(np.int32)
    cluster_sizes = np.bincount(labels, minlength=effective_clusters).astype(np.int64)
    quotas = compute_stratified_cluster_quotas(cluster_sizes, sample_count)

    rng = np.random.default_rng(seed)
    selected_ids: list[int] = []
    for cluster_id, quota in enumerate(quotas.tolist()):
        if quota <= 0:
            continue
        members = np.flatnonzero(labels == cluster_id)
        if int(members.size) <= quota:
            selected_ids.extend(int(x) for x in members.tolist())
            continue
        chosen = rng.choice(members, size=quota, replace=False)
        selected_ids.extend(int(x) for x in chosen.tolist())

    selected_ids = sorted(set(selected_ids))
    if len(selected_ids) > sample_count:
        selected_ids = selected_ids[:sample_count]
    elif len(selected_ids) < sample_count:
        remaining = sample_count - len(selected_ids)
        pool = np.setdiff1d(np.arange(total_vectors, dtype=np.int64), np.asarray(selected_ids, dtype=np.int64))
        if remaining > 0 and pool.size > 0:
            fill = rng.choice(pool, size=min(remaining, int(pool.size)), replace=False)
            selected_ids.extend(int(x) for x in fill.tolist())
            selected_ids.sort()
    return selected_ids, labels, effective_clusters


def write_queries_and_load_matrix(source, out_dir: Path) -> np.ndarray:
    if source.kind == "jsonl":
        assert source.queries_jsonl is not None
        queries_out = out_dir / "queries.jsonl"
        if not source.queries_jsonl.exists():
            return np.empty((0, source.vector_size), dtype=np.float32)
        matrix = load_queries_jsonl(source.queries_jsonl)
        with source.queries_jsonl.open("r", encoding="utf-8") as src, queries_out.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                dst.write(line if line.endswith("\n") else line + "\n")
        return matrix

    assert source.h5_path is not None
    queries_out = out_dir / "queries.jsonl"
    with h5py.File(source.h5_path, "r") as data, queries_out.open("w", encoding="utf-8") as dst:
        test = data["test"]
        matrix = np.asarray(test, dtype=np.float32)
        for i in range(test.shape[0]):
            dst.write(json.dumps(test[i].tolist()) + "\n")
    return matrix


def write_neighbors(
    sampled_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbours_path: Path,
    distance: str,
    top_k: int,
) -> int:
    if sampled_vectors.shape[0] == 0 or query_vectors.shape[0] == 0:
        return 0
    k = min(top_k, sampled_vectors.shape[0])
    metric = "cosine" if distance == "cosine" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(sampled_vectors)
    indices = nn.kneighbors(query_vectors, return_distance=False)
    with neighbours_path.open("w", encoding="utf-8") as fp:
        for row in indices:
            fp.write(json.dumps([int(x) for x in row.tolist()]) + "\n")
    return int(k)


def run(args: argparse.Namespace) -> None:
    datasets_root = Path(args.datasets_root).resolve()
    source = resolve_source(
        datasets_root=datasets_root,
        source_path=args.source_path,
        distance=args.distance,
        vector_size=args.vector_size,
    )

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.output_name
    sampled_dir = run_dir / "sampled"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    source_name = source.dataset_config.get("name") or sanitize_dataset_name(source.label)
    sampled_name = f"{args.dataset_name_prefix}-sampled"

    if source.kind == "jsonl":
        assert source.vectors_jsonl is not None
        total_vectors = count_train_vectors_jsonl(source.vectors_jsonl)
    else:
        assert source.h5_path is not None
        with h5py.File(source.h5_path, "r") as data:
            total_vectors = int(data["train"].shape[0])
    if total_vectors <= 0:
        raise RuntimeError("No train vectors found in source dataset.")

    sample_count = max(1, int(round(total_vectors * args.sample_ratio)))
    sample_count = min(sample_count, total_vectors)
    train_matrix = None
    sampling_meta: dict[str, object] = {"strategy": args.sampling_strategy}
    if args.sampling_strategy == "uniform":
        rng = np.random.default_rng(args.seed)
        selected_ids = sorted(rng.choice(total_vectors, size=sample_count, replace=False).tolist())
    elif args.sampling_strategy == "kmeans_stratified":
        train_matrix = load_train_matrix(source)
        if int(train_matrix.shape[0]) != total_vectors:
            raise RuntimeError(
                f"Unexpected train vector count mismatch: expected {total_vectors}, got {train_matrix.shape[0]}"
            )
        cluster_count = resolve_kmeans_clusters(args.kmeans_clusters, total_vectors)
        selected_ids, cluster_labels, actual_clusters = sample_ids_by_kmeans_stratified(
            train_matrix=train_matrix,
            sample_count=sample_count,
            seed=args.seed,
            n_clusters=cluster_count,
            batch_size=args.kmeans_batch_size,
        )
        sampling_meta.update(
            {
                "requested_clusters": int(cluster_count),
                "actual_clusters": int(actual_clusters),
                "cluster_size_min": int(np.bincount(cluster_labels, minlength=actual_clusters).min()),
                "cluster_size_max": int(np.bincount(cluster_labels, minlength=actual_clusters).max()),
            }
        )
    else:
        raise ValueError(f"Unsupported sampling strategy: {args.sampling_strategy}")
    selected_id_set = set(selected_ids)

    vectors_out = sampled_dir / "vectors.jsonl"
    payloads_out = sampled_dir / "payloads.jsonl"

    sampled_vectors: list[np.ndarray] = []
    payload_iter = source.iter_payloads()
    wrote_payloads = False

    with vectors_out.open("w", encoding="utf-8") as vf:
        pf = payloads_out.open("w", encoding="utf-8") if payload_iter is not None else None
        try:
            for idx, vector in source.iter_train_vectors():
                payload = next(payload_iter) if payload_iter is not None else None
                if idx not in selected_id_set:
                    continue
                vf.write(json.dumps(vector) + "\n")
                sampled_vectors.append(np.asarray(vector, dtype=np.float32))
                if pf is not None:
                    pf.write(json.dumps(payload) + "\n")
                    wrote_payloads = True
        finally:
            if pf is not None:
                pf.close()

    if not wrote_payloads and payloads_out.exists():
        payloads_out.unlink()

    sampled_matrix = np.vstack(sampled_vectors) if sampled_vectors else np.empty((0, source.vector_size), dtype=np.float32)
    query_matrix = write_queries_and_load_matrix(source, sampled_dir)

    top_k = args.neighbors_top_k
    if top_k <= 0:
        top_k = infer_top_k(source.source_dir, fallback=100)
    actual_top_k = write_neighbors(
        sampled_vectors=sampled_matrix,
        query_vectors=query_matrix,
        neighbours_path=sampled_dir / "neighbours.jsonl",
        distance=source.distance,
        top_k=top_k,
    )

    info = {
        "source_kind": source.kind,
        "source_label": source.label,
        "source_dataset_name": source_name,
        "source_dir": str(source.source_dir),
        "sampled_dir": str(sampled_dir),
        "sampled_dataset_name": sampled_name,
        "sample_ratio": args.sample_ratio,
        "sampling_strategy": args.sampling_strategy,
        "sampling_meta": sampling_meta,
        "seed": args.seed,
        "vector_size": source.vector_size,
        "distance": source.distance,
        "stats": {
            "total_vectors": total_vectors,
            "sampled_vectors": sample_count,
            "sample_ratio_actual": sample_count / total_vectors,
            "queries": int(query_matrix.shape[0]),
            "neighbors_top_k": int(actual_top_k),
        },
    }
    info_path = run_dir / "sample_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "sample_info": str(info_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a dataset's base vectors while keeping query vectors unchanged."
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory containing benchmark datasets.",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Dataset name (from datasets.json), folder path, or .hdf5 file path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for sampled datasets.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Run folder name under output-root.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        required=True,
        help="Prefix used to name sampled dataset for benchmark outputs.",
    )
    parser.add_argument("--sample-ratio", type=float, required=True)
    parser.add_argument(
        "--sampling-strategy",
        choices=["uniform", "kmeans_stratified"],
        default="kmeans_stratified",
        help="Sampling strategy: global uniform or k-means stratified by cluster size.",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=0,
        help="Number of k-means clusters for stratified sampling; <=0 uses heuristic sqrt(N) capped at 100.",
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=2048,
        help="MiniBatchKMeans batch size for stratified sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distance",
        default="",
        help="Distance metric; empty means use datasets.json or cosine.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=0,
        help="Vector dimension; 0 means auto-detect.",
    )
    parser.add_argument(
        "--neighbors-top-k",
        type=int,
        default=0,
        help="Ground-truth neighbors per query; <=0 means infer from source or use 100.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
