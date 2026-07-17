#!/usr/bin/env python3
"""Generate initial/drift-increment/post-drift datasets via cluster-aware sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np


def iter_jsonl(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_datasets_registry(datasets_root: Path) -> dict[str, dict]:
    config_path = datasets_root / "datasets.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fp:
        entries = json.load(fp)
    return {entry["name"]: entry for entry in entries}


def kmeans(data: np.ndarray, k: int, seed: int, max_iter: int = 30) -> np.ndarray:
    if data.shape[0] < k:
        raise ValueError(f"Sample size ({data.shape[0]}) must be >= n_clusters ({k}).")
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(data.shape[0], size=k, replace=False)].copy()

    for _ in range(max_iter):
        dist = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            points = data[labels == i]
            if points.size == 0:
                new_centroids[i] = data[rng.integers(0, data.shape[0])]
            else:
                new_centroids[i] = points.mean(axis=0)
        if np.allclose(centroids, new_centroids, rtol=1e-4, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


@dataclass
class SplitStats:
    total_vectors: int = 0
    initial_vectors: int = 0
    drift_increment_vectors: int = 0
    post_drift_vectors: int = 0
    base_cluster_points: int = 0
    drift_cluster_points: int = 0


@dataclass
class DatasetSource:
    kind: str  # "jsonl" | "h5" | "npy"
    label: str
    vector_size: int
    distance: str
    source_dir: Path
    vectors_jsonl: Optional[Path] = None
    payloads_jsonl: Optional[Path] = None
    queries_jsonl: Optional[Path] = None
    neighbours_jsonl: Optional[Path] = None
    h5_path: Optional[Path] = None
    vectors_npy: Optional[Path] = None
    tests_jsonl: Optional[Path] = None
    dataset_config: dict = field(default_factory=dict)

    def iter_train_vectors(self) -> Iterator[tuple[int, list]]:
        if self.kind == "jsonl":
            assert self.vectors_jsonl is not None
            for idx, vector in enumerate(iter_jsonl(self.vectors_jsonl)):
                yield idx, vector
            return
        if self.kind == "npy":
            assert self.vectors_npy is not None
            vectors = np.load(self.vectors_npy, mmap_mode="r")
            for idx in range(vectors.shape[0]):
                yield idx, vectors[idx].tolist()
            return
        assert self.h5_path is not None
        with h5py.File(self.h5_path, "r") as data:
            train = data["train"]
            for idx in range(train.shape[0]):
                yield idx, train[idx].tolist()

    def reservoir_sample(self, sample_size: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sample: list[np.ndarray] = []
        for i, (_, vec) in enumerate(self.iter_train_vectors()):
            arr = np.asarray(vec, dtype=np.float32)
            if i < sample_size:
                sample.append(arr)
            else:
                j = rng.integers(0, i + 1)
                if j < sample_size:
                    sample[int(j)] = arr
        if not sample:
            raise ValueError(f"No vectors found in source: {self.label}")
        return np.vstack(sample)

    def iter_payloads(self) -> Optional[Iterator[object]]:
        if self.payloads_jsonl is None or not self.payloads_jsonl.exists():
            return None
        return iter_jsonl(self.payloads_jsonl)

    def _write_query_files_jsonl(self, out_dir: Path, idx_map: dict[int, int]) -> None:
        if self.queries_jsonl is None or not self.queries_jsonl.exists():
            return
        queries_out = out_dir / "queries.jsonl"
        neighbours_out = out_dir / "neighbours.jsonl"
        with self.queries_jsonl.open("r", encoding="utf-8") as qf, queries_out.open(
            "w", encoding="utf-8"
        ) as qout:
            if self.neighbours_jsonl is not None and self.neighbours_jsonl.exists():
                with self.neighbours_jsonl.open("r", encoding="utf-8") as nf, neighbours_out.open(
                    "w", encoding="utf-8"
                ) as nout:
                    for qline, nline in zip(qf, nf):
                        qout.write(qline if qline.endswith("\n") else qline + "\n")
                        neighbors = json.loads(nline)
                        remapped = [idx_map[int(n)] for n in neighbors if int(n) in idx_map]
                        nout.write(json.dumps(remapped) + "\n")
            else:
                for qline in qf:
                    qout.write(qline if qline.endswith("\n") else qline + "\n")

    def _write_query_files_h5(self, out_dir: Path, idx_map: dict[int, int]) -> None:
        assert self.h5_path is not None
        queries_out = out_dir / "queries.jsonl"
        neighbours_out = out_dir / "neighbours.jsonl"
        with h5py.File(self.h5_path, "r") as data, queries_out.open(
            "w", encoding="utf-8"
        ) as qout, neighbours_out.open("w", encoding="utf-8") as nout:
            test = data["test"]
            neighbors = data["neighbors"]
            for i in range(test.shape[0]):
                qout.write(json.dumps(test[i].tolist()) + "\n")
                remapped = [idx_map[int(n)] for n in neighbors[i] if int(n) in idx_map]
                nout.write(json.dumps(remapped) + "\n")

    def _write_query_files_npy(self, out_dir: Path, idx_map: dict[int, int]) -> None:
        if self.tests_jsonl is None or not self.tests_jsonl.exists():
            return
        queries_out = out_dir / "queries.jsonl"
        neighbours_out = out_dir / "neighbours.jsonl"
        with self.tests_jsonl.open("r", encoding="utf-8") as tf, queries_out.open(
            "w", encoding="utf-8"
        ) as qout, neighbours_out.open("w", encoding="utf-8") as nout:
            for line in tf:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                query = row.get("query")
                if query is None:
                    continue
                qout.write(json.dumps(query) + "\n")
                neighbors = row.get("closest_ids") or []
                remapped = [idx_map[int(n)] for n in neighbors if int(n) in idx_map]
                nout.write(json.dumps(remapped) + "\n")

    def write_query_files(self, out_dir: Path, idx_map: dict[int, int]) -> None:
        if self.kind == "jsonl":
            self._write_query_files_jsonl(out_dir, idx_map)
        elif self.kind == "npy":
            self._write_query_files_npy(out_dir, idx_map)
        else:
            self._write_query_files_h5(out_dir, idx_map)


def resolve_source(
    datasets_root: Path,
    source_path: str,
    distance: str,
    vector_size: int,
) -> DatasetSource:
    datasets_root = datasets_root.resolve()
    registry = load_datasets_registry(datasets_root)
    config = registry.get(source_path)
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = datasets_root / candidate

    if config:
        candidate = datasets_root / config["path"]
        kind = config.get("type", "jsonl")
        if not distance and config.get("distance"):
            distance = config["distance"]
        if vector_size == 0:
            vector_size = int(config["vector_size"])
        label = config["name"]
    else:
        kind = None
        label = source_path

    if candidate.is_file():
        if candidate.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Unsupported source file: {candidate}")
        kind = "h5"
        h5_path = candidate
        source_dir = candidate.parent
    elif candidate.is_dir():
        source_dir = candidate
        h5_files = sorted(candidate.glob("*.hdf5")) + sorted(candidate.glob("*.h5"))
        vectors_npy = candidate / "vectors.npy"
        tests_jsonl = candidate / "tests.jsonl"
        if (candidate / "vectors.jsonl").exists():
            kind = kind or "jsonl"
        elif h5_files:
            kind = "h5"
            h5_path = h5_files[0]
        elif vectors_npy.exists():
            kind = "npy"
        else:
            raise FileNotFoundError(f"No vectors.jsonl/.hdf5/vectors.npy found in: {candidate}")
    else:
        raise FileNotFoundError(f"Source path does not exist: {candidate}")

    if kind == "jsonl":
        vectors_jsonl = source_dir / "vectors.jsonl"
        if not vectors_jsonl.exists():
            raise FileNotFoundError(f"vectors.jsonl not found in: {source_dir}")
        if vector_size == 0:
            first = next(iter_jsonl(vectors_jsonl))
            vector_size = len(first)  # type: ignore[arg-type]
        return DatasetSource(
            kind="jsonl",
            label=label,
            vector_size=vector_size,
            distance=distance or "cosine",
            source_dir=source_dir,
            vectors_jsonl=vectors_jsonl,
            payloads_jsonl=source_dir / "payloads.jsonl",
            queries_jsonl=source_dir / "queries.jsonl",
            neighbours_jsonl=source_dir / "neighbours.jsonl",
            dataset_config=config or {},
        )

    if kind == "h5":
        if "h5_path" not in locals():
            h5_path = candidate if candidate.is_file() else h5_files[0]
        with h5py.File(h5_path, "r") as data:
            if vector_size == 0:
                vector_size = int(data["train"].shape[1])
        return DatasetSource(
            kind="h5",
            label=label,
            vector_size=vector_size,
            distance=distance or "cosine",
            source_dir=source_dir,
            h5_path=h5_path.resolve(),
            dataset_config=config or {},
        )

    if kind == "npy":
        if "vectors_npy" not in locals():
            vectors_npy = source_dir / "vectors.npy"
        if vector_size == 0:
            vectors = np.load(vectors_npy, mmap_mode="r")
            vector_size = int(vectors.shape[1])
        return DatasetSource(
            kind="npy",
            label=label,
            vector_size=vector_size,
            distance=distance or "cosine",
            source_dir=source_dir,
            vectors_npy=vectors_npy.resolve(),
            tests_jsonl=(source_dir / "tests.jsonl").resolve() if (source_dir / "tests.jsonl").exists() else None,
            dataset_config=config or {},
        )

    raise ValueError(f"Unsupported dataset type: {kind}")


def write_dataset_info(
    info_path: Path,
    source: DatasetSource,
    initial_dir: Path,
    drift_increment_dir: Path,
    post_drift_dir: Path,
    dataset_name_prefix: str,
    stats: SplitStats,
    n_clusters: int,
    m_clusters: int,
    base_ratio: float,
    drift_ratio: float,
    drift_cluster_ids: list[int],
    seed: int,
) -> None:
    data = {
        "source_kind": source.kind,
        "source_label": source.label,
        "source_dir": str(source.source_dir),
        "source_h5": str(source.h5_path) if source.h5_path else None,
        "initial_dir": str(initial_dir),
        "drift_increment_dir": str(drift_increment_dir),
        "post_drift_dir": str(post_drift_dir),
        "dataset_name_initial": f"{dataset_name_prefix}-initial",
        "dataset_name_drift_increment": f"{dataset_name_prefix}-drift-increment",
        "dataset_name_post_drift": f"{dataset_name_prefix}-post-drift",
        "n_clusters": n_clusters,
        "m_clusters": m_clusters,
        "base_cluster_initial_ratio": base_ratio,
        "drift_cluster_initial_ratio": drift_ratio,
        "drift_cluster_ids": drift_cluster_ids,
        "seed": seed,
        "vector_size": source.vector_size,
        "distance": source.distance,
        "stats": {
            "total_vectors": stats.total_vectors,
            "initial_vectors": stats.initial_vectors,
            "drift_increment_vectors": stats.drift_increment_vectors,
            "post_drift_vectors": stats.post_drift_vectors,
            "base_cluster_points": stats.base_cluster_points,
            "drift_cluster_points": stats.drift_cluster_points,
            "initial_ratio_actual": (stats.initial_vectors / stats.total_vectors)
            if stats.total_vectors
            else 0.0,
            "drift_increment_ratio_actual": (
                stats.drift_increment_vectors / stats.total_vectors
            )
            if stats.total_vectors
            else 0.0,
        },
    }
    info_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    datasets_root = Path(args.datasets_root).resolve()
    source = resolve_source(
        datasets_root=datasets_root,
        source_path=args.source_path,
        distance=args.distance,
        vector_size=args.vector_size,
    )

    if args.m_clusters <= 0 or args.m_clusters >= args.n_clusters:
        raise ValueError("m_clusters must satisfy: 0 < m_clusters < n_clusters")
    if not (0.0 <= args.base_cluster_initial_ratio <= 1.0):
        raise ValueError("base_cluster_initial_ratio must be in [0,1]")
    if not (0.0 <= args.drift_cluster_initial_ratio <= 1.0):
        raise ValueError("drift_cluster_initial_ratio must be in [0,1]")

    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.output_name
    initial_dir = run_dir / "initial"
    drift_increment_dir = run_dir / "drift_increment"
    post_drift_dir = run_dir / "post_drift"
    initial_dir.mkdir(parents=True, exist_ok=True)
    drift_increment_dir.mkdir(parents=True, exist_ok=True)
    post_drift_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Source: {source.label} ({source.kind}), "
        f"vectors={source.vector_size}d, distance={source.distance}"
    )

    sample = source.reservoir_sample(args.max_kmeans_sample, args.seed)
    centroids = kmeans(sample, args.n_clusters, args.seed, args.kmeans_max_iter)
    rng = np.random.default_rng(args.seed)
    drift_clusters = sorted(
        rng.choice(np.arange(args.n_clusters), size=args.m_clusters, replace=False).tolist()
    )
    drift_cluster_set = set(drift_clusters)

    stats = SplitStats()
    map_initial: dict[int, int] = {}
    map_drift_increment: dict[int, int] = {}
    map_post_drift: dict[int, int] = {}

    init_vectors_fp = (initial_dir / "vectors.jsonl").open("w", encoding="utf-8")
    drift_vectors_fp = (drift_increment_dir / "vectors.jsonl").open("w", encoding="utf-8")
    post_vectors_fp = (post_drift_dir / "vectors.jsonl").open("w", encoding="utf-8")

    init_payloads_fp: Optional[object] = None
    drift_payloads_fp: Optional[object] = None
    post_payloads_fp: Optional[object] = None

    try:
        payload_iter = source.iter_payloads()
        if payload_iter is not None:
            init_payloads_fp = (initial_dir / "payloads.jsonl").open("w", encoding="utf-8")
            drift_payloads_fp = (drift_increment_dir / "payloads.jsonl").open(
                "w", encoding="utf-8"
            )
            post_payloads_fp = (post_drift_dir / "payloads.jsonl").open("w", encoding="utf-8")

        for old_idx, vector in source.iter_train_vectors():
            arr = np.asarray(vector, dtype=np.float32)
            cluster = int(np.argmin(np.linalg.norm(centroids - arr, axis=1)))
            is_drift_cluster = cluster in drift_cluster_set
            if is_drift_cluster:
                stats.drift_cluster_points += 1
                p_initial = args.drift_cluster_initial_ratio
            else:
                stats.base_cluster_points += 1
                p_initial = args.base_cluster_initial_ratio

            assign_to_initial = bool(rng.random() < p_initial)
            payload = next(payload_iter) if payload_iter is not None else None

            if assign_to_initial:
                map_initial[old_idx] = stats.initial_vectors
                init_vectors_fp.write(json.dumps(vector) + "\n")
                if init_payloads_fp is not None:
                    init_payloads_fp.write(json.dumps(payload) + "\n")
                stats.initial_vectors += 1
            else:
                map_drift_increment[old_idx] = stats.drift_increment_vectors
                drift_vectors_fp.write(json.dumps(vector) + "\n")
                if drift_payloads_fp is not None:
                    drift_payloads_fp.write(json.dumps(payload) + "\n")
                stats.drift_increment_vectors += 1

            map_post_drift[old_idx] = stats.post_drift_vectors
            post_vectors_fp.write(json.dumps(vector) + "\n")
            if post_payloads_fp is not None:
                post_payloads_fp.write(json.dumps(payload) + "\n")
            stats.post_drift_vectors += 1
            stats.total_vectors += 1

            if stats.total_vectors % 100000 == 0:
                print(f"  processed {stats.total_vectors} vectors...")
    finally:
        init_vectors_fp.close()
        drift_vectors_fp.close()
        post_vectors_fp.close()
        if init_payloads_fp is not None:
            init_payloads_fp.close()
        if drift_payloads_fp is not None:
            drift_payloads_fp.close()
        if post_payloads_fp is not None:
            post_payloads_fp.close()

    if stats.initial_vectors == 0 or stats.drift_increment_vectors == 0:
        raise RuntimeError(
            "Degenerated split encountered (one side is empty). Adjust ratios, n/m, or seed."
        )

    print("Exporting queries with remapped neighbour indices...")
    source.write_query_files(initial_dir, map_initial)
    source.write_query_files(drift_increment_dir, map_drift_increment)
    source.write_query_files(post_drift_dir, map_post_drift)

    info_path = run_dir / "drift_info.json"
    write_dataset_info(
        info_path=info_path,
        source=source,
        initial_dir=initial_dir,
        drift_increment_dir=drift_increment_dir,
        post_drift_dir=post_drift_dir,
        dataset_name_prefix=args.dataset_name_prefix,
        stats=stats,
        n_clusters=args.n_clusters,
        m_clusters=args.m_clusters,
        base_ratio=args.base_cluster_initial_ratio,
        drift_ratio=args.drift_cluster_initial_ratio,
        drift_cluster_ids=drift_clusters,
        seed=args.seed,
    )

    print(json.dumps({"ok": True, "drift_info": str(info_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cluster-based drift datasets.")
    parser.add_argument(
        "--datasets-root",
        default="/path/to/TransTune/vector-db-benchmark-master/datasets",
        help="Root directory containing benchmark datasets.",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Dataset name (from datasets.json), folder path, or .hdf5 file path.",
    )
    parser.add_argument(
        "--output-root",
        default="/path/to/TransTune/vector-db-benchmark-master/datasets/new_adapt",
        help="Output root for generated initial/drift/post_drift datasets.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Run folder name under output-root.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        required=True,
        help="Prefix used to name generated datasets for benchmark outputs.",
    )
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--m-clusters", type=int, default=2)
    parser.add_argument("--base-cluster-initial-ratio", type=float, default=0.8)
    parser.add_argument("--drift-cluster-initial-ratio", type=float, default=0.2)
    parser.add_argument("--max-kmeans-sample", type=int, default=50000)
    parser.add_argument("--kmeans-max-iter", type=int, default=30)
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
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
