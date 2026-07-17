#!/usr/bin/env python3
"""End-to-end drift adaptive tuning pipeline.

Pipeline:
1) Evaluate baseline performance on original dataset with initial index config.
2) Simulate data drift by insert/delete vectors (random or skewed by clusters).
3) On drifted dataset: run sampled index-parameter sweep, pick sampled top-K.
4) Re-test sampled top-K configs on full drifted dataset.
5) Output final best top-3 configs on drifted full dataset and compare to baseline.

Usage:
- Edit CONFIG in this file.
- Run: python3 auto-configure/transtune/new_adapt/run_drift_adaptive_pipeline.py
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Limit BLAS/OpenMP threads early to avoid OpenBLAS NUM_THREADS crashes
# on high-core machines, especially when background/nohup runs fan out work.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from generate_drift_dataset import resolve_source


# ===================== User Config =====================
CONFIG: dict[str, Any] = {
    # Source dataset: name in datasets.json, folder path, or .h5/.hdf5 path
    "source_path": "glove-100-angular",
    "distance": "",  # empty => auto resolve
    "vector_size": 0,  # 0 => auto resolve
    "seed": 42,
    "neighbors_top_k": 0,  # <=0 => infer from source neighbours or fallback(100)

    # Benchmark engine settings
    "engine_name": "milvus-p10",
    "host": "127.0.0.1",
    "config_json": "",  # empty => default milvus-single-node config

    # Baseline initial index config (on original dataset)
    "initial_index_type": "SCANN",
    "initial_index_params": {"nlist": 1942},
    "initial_search_params": {"nprobe": 190, "reorder_k": 167},

    # Drift simulation
    # drift_mode: "random" | "skewed"
    "drift_mode": "skewed",
    "delete_count": 20000,
    "insert_count": 100000,
    "insert_noise_std": 0.01,
    # skewed mode params
    "skewed_n_clusters": 20,
    "skewed_m_hot_clusters": 4,
    "skewed_hot_fraction": 0.9,  # fraction of insert/delete applied to hot clusters
    "skewed_cold_fraction": 0.1,  # remaining clusters
    "skewed_kmeans_batch_size": 4096,

    # Sampled sweep on drifted dataset (logic equivalent to run_sampling_index_sweep.py)
    "sample_ratio": 0.1,
    "sampling_strategy": "uniform",  # uniform | kmeans_stratified
    "sampling_kmeans_clusters": 50,
    "sampling_kmeans_batch_size": 2048,
    "sweep_index_type": "SCANN",
    "sweep_index_param_grid": {"nlist": [1000, 1250, 1500, 1750, 2000, 2500, 2750,3000]},
    "sweep_search_param_grid": {"nprobe": [100, 150, 200, 250, 300, 350], "reorder_k": [101, 150, 200, 250]},
    "sampled_recall_threshold": 0.8,
    "sampled_top_k": 5,
    "continue_on_sweep_error": True,

    # Full drifted dataset evaluation (logic equivalent to run_full_dataset_topn_eval.py)
    "full_best_metric": "rps",  # "rps" | "p95_time"
    "full_recall_threshold": 0.0,  # <=0 means disabled

    # Final output count
    "final_top_n": 3,

    # Optional run tag
    "run_tag": "",  # empty => auto generated
}
# =======================================================


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "vector-db-benchmark-master"
DEFAULT_DATASETS_ROOT = BENCHMARK_ROOT / "datasets"
DEFAULT_NEW_ADAPT_ROOT = DEFAULT_DATASETS_ROOT / "new_adapt"
DEFAULT_CONFIG_JSON = (
    BENCHMARK_ROOT
    / "experiments"
    / "configurations"
    / "milvus-single-node.json"
)


@dataclass
class Perf:
    rps: float | None
    p95_time: float | None
    recall: float | None
    result_file: str


@dataclass
class SweepRow:
    combo_id: int
    index_type: str
    index_params: dict[str, Any]
    search_params: dict[str, Any]
    rps: float | None
    p95_time: float | None
    recall: float | None
    result_file: str
    status: str
    error: str


@dataclass
class FullEvalRow:
    sampled_rank: int
    sampled_combo_id: int
    index_type: str
    index_params: dict[str, Any]
    search_params: dict[str, Any]
    sampled_rps: float | None
    sampled_p95_time: float | None
    sampled_recall: float | None
    full_rps: float | None
    full_p95_time: float | None
    full_recall: float | None
    full_result_file: str
    status: str
    error: str


@dataclass
class StageTiming:
    name: str
    seconds: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def infer_top_k_for_source(source) -> int:
    if source.kind == "h5":
        assert source.h5_path is not None
        with h5py.File(source.h5_path, "r") as data:
            if "neighbors" in data and len(data["neighbors"].shape) == 2:
                return int(data["neighbors"].shape[1])
        return 100
    if source.kind == "npy":
        tests_path = source.source_dir / "tests.jsonl"
        if tests_path.exists():
            with tests_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    closest_ids = row.get("closest_ids")
                    if isinstance(closest_ids, list) and closest_ids:
                        return len(closest_ids)
        return 100
    npath = source.source_dir / "neighbours.jsonl"
    if not npath.exists():
        return 100
    with npath.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                vals = json.loads(line)
                if isinstance(vals, list) and vals:
                    return len(vals)
    return 100


def load_queries_from_source(source) -> np.ndarray:
    if source.kind == "h5":
        assert source.h5_path is not None
        with h5py.File(source.h5_path, "r") as data:
            return np.asarray(data["test"], dtype=np.float32)
    if source.kind == "npy":
        tests_path = source.source_dir / "tests.jsonl"
        if not tests_path.exists():
            return np.empty((0, source.vector_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        with tests_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                query = row.get("query")
                if query is not None:
                    rows.append(np.asarray(query, dtype=np.float32))
        if not rows:
            return np.empty((0, source.vector_size), dtype=np.float32)
        return np.vstack(rows)
    qpath = source.source_dir / "queries.jsonl"
    if not qpath.exists():
        return np.empty((0, source.vector_size), dtype=np.float32)
    rows: list[np.ndarray] = []
    with qpath.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(np.asarray(json.loads(line), dtype=np.float32))
    if not rows:
        return np.empty((0, source.vector_size), dtype=np.float32)
    return np.vstack(rows)


def load_train_vectors_and_payloads(source) -> tuple[np.ndarray, list[Any] | None]:
    vectors: list[np.ndarray] = []
    payloads: list[Any] | None = [] if source.iter_payloads() is not None else None
    payload_iter = source.iter_payloads()
    for _, vec in source.iter_train_vectors():
        vectors.append(np.asarray(vec, dtype=np.float32))
        if payloads is not None:
            payloads.append(next(payload_iter))  # type: ignore[arg-type]
    if not vectors:
        return np.empty((0, source.vector_size), dtype=np.float32), payloads
    return np.vstack(vectors), payloads


def write_jsonl_vectors(path: Path, vectors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for i in range(vectors.shape[0]):
            fp.write(json.dumps(vectors[i].tolist()) + "\n")


def write_jsonl_payloads(path: Path, payloads: list[Any] | None) -> None:
    if payloads is None:
        return
    with path.open("w", encoding="utf-8") as fp:
        for payload in payloads:
            fp.write(json.dumps(payload) + "\n")


def write_jsonl_queries(path: Path, queries: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for i in range(queries.shape[0]):
            fp.write(json.dumps(queries[i].tolist()) + "\n")


def recompute_neighbors(
    vectors: np.ndarray,
    queries: np.ndarray,
    distance: str,
    top_k: int,
    out_path: Path,
) -> int:
    if vectors.shape[0] == 0 or queries.shape[0] == 0:
        out_path.write_text("", encoding="utf-8")
        return 0
    k = min(int(top_k), int(vectors.shape[0]))
    metric = "cosine" if distance == "cosine" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(vectors)
    indices = nn.kneighbors(queries, return_distance=False)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in indices:
            fp.write(json.dumps([int(x) for x in row.tolist()]) + "\n")
    return k


def normalize_weights(hot_fraction: float, cold_fraction: float) -> tuple[float, float]:
    hot = max(0.0, float(hot_fraction))
    cold = max(0.0, float(cold_fraction))
    s = hot + cold
    if s <= 0:
        return 0.5, 0.5
    return hot / s, cold / s


def sample_indices_from_pool(
    rng: np.random.Generator,
    pool: np.ndarray,
    count: int,
    replace: bool = False,
) -> np.ndarray:
    if count <= 0 or pool.size == 0:
        return np.empty((0,), dtype=np.int64)
    if not replace and count >= int(pool.size):
        return pool.copy()
    return np.asarray(
        rng.choice(pool, size=count, replace=replace),
        dtype=np.int64,
    )


def simulate_drift(
    vectors: np.ndarray,
    payloads: list[Any] | None,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[Any] | None, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["seed"]))
    n = int(vectors.shape[0])
    delete_count = max(0, min(int(cfg["delete_count"]), n))
    insert_count = max(0, int(cfg["insert_count"]))
    mode = str(cfg["drift_mode"])
    noise_std = float(cfg["insert_noise_std"])

    if n == 0:
        return vectors, payloads, {"mode": mode, "delete_count": 0, "insert_count": 0}

    delete_ids: np.ndarray
    insert_base_ids: np.ndarray
    skew_meta: dict[str, Any] = {}

    if mode == "skewed":
        n_clusters = int(cfg["skewed_n_clusters"])
        m_hot = int(cfg["skewed_m_hot_clusters"])
        if n_clusters <= 1:
            raise ValueError("skewed_n_clusters must be > 1 for skewed drift.")
        if m_hot <= 0 or m_hot >= n_clusters:
            raise ValueError("skewed_m_hot_clusters must satisfy 0 < M < n_clusters.")
        kmeans = MiniBatchKMeans(
            n_clusters=min(n_clusters, n),
            random_state=int(cfg["seed"]),
            batch_size=max(256, int(cfg["skewed_kmeans_batch_size"])),
            n_init="auto",
        )
        labels = kmeans.fit_predict(vectors).astype(np.int32)
        actual_clusters = int(labels.max()) + 1
        if m_hot >= actual_clusters:
            m_hot = max(1, actual_clusters - 1)
        hot_clusters = sorted(
            rng.choice(np.arange(actual_clusters), size=m_hot, replace=False).tolist()
        )
        hot_set = set(hot_clusters)
        hot_idx = np.flatnonzero(np.isin(labels, hot_clusters))
        cold_idx = np.flatnonzero(~np.isin(labels, hot_clusters))

        hot_w, cold_w = normalize_weights(
            float(cfg["skewed_hot_fraction"]),
            float(cfg["skewed_cold_fraction"]),
        )

        delete_hot = min(int(round(delete_count * hot_w)), int(hot_idx.size))
        delete_cold = min(delete_count - delete_hot, int(cold_idx.size))
        delete_ids = np.concatenate(
            [
                sample_indices_from_pool(rng, hot_idx, delete_hot, replace=False),
                sample_indices_from_pool(rng, cold_idx, delete_cold, replace=False),
            ]
        )
        if int(delete_ids.size) < delete_count:
            remaining_pool = np.setdiff1d(np.arange(n, dtype=np.int64), delete_ids)
            extra = sample_indices_from_pool(
                rng, remaining_pool, delete_count - int(delete_ids.size), replace=False
            )
            delete_ids = np.concatenate([delete_ids, extra])

        insert_hot = int(round(insert_count * hot_w))
        insert_cold = insert_count - insert_hot
        hot_for_insert = hot_idx if hot_idx.size > 0 else np.arange(n, dtype=np.int64)
        cold_for_insert = cold_idx if cold_idx.size > 0 else np.arange(n, dtype=np.int64)
        insert_base_ids = np.concatenate(
            [
                sample_indices_from_pool(rng, hot_for_insert, insert_hot, replace=True),
                sample_indices_from_pool(rng, cold_for_insert, insert_cold, replace=True),
            ]
        )
        skew_meta = {
            "actual_clusters": actual_clusters,
            "hot_clusters": hot_clusters,
            "hot_cluster_count": int(hot_idx.size),
            "cold_cluster_count": int(cold_idx.size),
            "hot_fraction_effective": hot_w,
            "cold_fraction_effective": cold_w,
        }
    else:
        delete_ids = sample_indices_from_pool(
            rng, np.arange(n, dtype=np.int64), delete_count, replace=False
        )
        insert_base_ids = sample_indices_from_pool(
            rng, np.arange(n, dtype=np.int64), insert_count, replace=True
        )

    delete_ids = np.unique(delete_ids.astype(np.int64))
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[delete_ids] = False
    kept_vectors = vectors[keep_mask]
    kept_payloads: list[Any] | None = None
    if payloads is not None:
        kept_payloads = [payloads[i] for i in np.flatnonzero(keep_mask).tolist()]

    if insert_base_ids.size > 0:
        base = vectors[insert_base_ids]
        noise = rng.normal(0.0, noise_std, size=base.shape).astype(np.float32)
        inserted_vectors = (base + noise).astype(np.float32)
        drifted_vectors = np.vstack([kept_vectors, inserted_vectors])
        if kept_payloads is not None and payloads is not None:
            inserted_payloads = [deepcopy(payloads[int(i)]) for i in insert_base_ids.tolist()]
            kept_payloads.extend(inserted_payloads)
    else:
        drifted_vectors = kept_vectors

    meta = {
        "mode": mode,
        "seed": int(cfg["seed"]),
        "original_vectors": n,
        "delete_count_requested": int(cfg["delete_count"]),
        "delete_count_applied": int(delete_ids.size),
        "insert_count_requested": int(cfg["insert_count"]),
        "insert_count_applied": int(insert_base_ids.size),
        "insert_noise_std": noise_std,
        "drifted_vectors": int(drifted_vectors.shape[0]),
        "delete_ratio": float(delete_ids.size) / float(n),
        "insert_ratio": float(insert_base_ids.size) / float(n),
        "skewed_meta": skew_meta,
    }
    return drifted_vectors, kept_payloads, meta


def get_dataset_for_benchmark(source, cfg: dict[str, Any], run_tag: str) -> dict[str, Any]:
    if source.kind == "jsonl":
        return {
            "dataset_name": f"new-adapt-{run_tag}-source",
            "dataset_path": str(source.source_dir),
            "vector_size": int(source.vector_size),
            "distance": str(source.distance),
        }

    # non-jsonl: if source comes from datasets.json, benchmark can resolve by name
    dataset_name = str(source.dataset_config.get("name") or "")
    if not dataset_name:
        raise ValueError(
            "For non-jsonl source without datasets.json registration, baseline benchmark is unsupported. "
            "Please use dataset name in datasets.json or jsonl source."
        )
    return {
        "dataset_name": dataset_name,
        "dataset_path": "",
        "vector_size": int(source.vector_size),
        "distance": str(source.distance),
    }


def set_engine_params(
    engine_cfg: dict[str, Any],
    index_type: str,
    index_params: dict[str, Any],
    search_params: dict[str, Any],
) -> None:
    upload = engine_cfg.setdefault("upload_params", {})
    upload["index_type"] = str(index_type)
    upload["index_params"] = dict(index_params)

    search_cfg = engine_cfg.get("search_params")
    if not isinstance(search_cfg, list) or not search_cfg:
        raise ValueError("search_params must be a non-empty list in engine config.")
    for item in search_cfg:
        item["params"] = dict(search_params)


def run_benchmark_once(
    engine_name: str,
    host: str,
    dataset_name: str,
    dataset_path: str,
    vector_size: int,
    distance: str,
    result_meta_path: Path,
) -> Perf:
    cmd = [
        "python3",
        str(SCRIPT_DIR / "run_custom_benchmark.py"),
        "--benchmark-root",
        str(BENCHMARK_ROOT),
        "--engine-name",
        engine_name,
        "--dataset-name",
        dataset_name,
        "--vector-size",
        str(vector_size),
        "--distance",
        distance,
        "--host",
        host,
        "--result-json",
        str(result_meta_path),
    ]
    if dataset_path.strip():
        cmd.extend(["--dataset-path", dataset_path])
    run_cmd(cmd, cwd=SCRIPT_DIR)

    meta = load_json(result_meta_path)
    result_file = Path(meta["result_file"]).resolve()
    payload = load_json(result_file)
    results = payload.get("results", {})
    return Perf(
        rps=float(results["rps"]) if results.get("rps") is not None else None,
        p95_time=float(results["p95_time"]) if results.get("p95_time") is not None else None,
        recall=float(results["mean_precisions"]) if results.get("mean_precisions") is not None else None,
        result_file=str(result_file),
    )


def cartesian_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    return [{k: v for k, v in zip(keys, item)} for item in itertools.product(*values)]


def normalize_grid(grid: dict[str, Any], name: str) -> dict[str, list[Any]]:
    if not isinstance(grid, dict):
        raise ValueError(f"{name} must be dict.")
    out: dict[str, list[Any]] = {}
    for k, v in grid.items():
        vals = v if isinstance(v, list) else [v]
        if not vals:
            raise ValueError(f"{name}.{k} cannot be empty.")
        out[str(k)] = vals
    return out


def run_sampling_generation_for_drifted(
    drifted_dataset_dir: Path,
    source_distance: str,
    source_vector_size: int,
    cfg: dict[str, Any],
    output_name: str,
) -> dict[str, Any]:
    cmd = [
        "python3",
        str(SCRIPT_DIR / "sampling" / "generate_sampled_dataset.py"),
        "--source-path",
        str(drifted_dataset_dir),
        "--sample-ratio",
        str(cfg["sample_ratio"]),
        "--sampling-strategy",
        str(cfg["sampling_strategy"]),
        "--kmeans-clusters",
        str(cfg["sampling_kmeans_clusters"]),
        "--kmeans-batch-size",
        str(cfg["sampling_kmeans_batch_size"]),
        "--seed",
        str(cfg["seed"]),
        "--distance",
        source_distance,
        "--vector-size",
        str(source_vector_size),
        "--neighbors-top-k",
        str(cfg["neighbors_top_k"]),
        "--output-name",
        output_name,
        "--dataset-name-prefix",
        f"new-adapt-{output_name}",
    ]
    run_cmd(cmd, cwd=SCRIPT_DIR / "sampling")
    sample_info_path = DEFAULT_NEW_ADAPT_ROOT / output_name / "sample_info.json"
    if not sample_info_path.exists():
        raise FileNotFoundError(f"sample_info.json not found: {sample_info_path}")
    return load_json(sample_info_path)


def choose_sampled_topk(rows: list[SweepRow], recall_threshold: float, top_k: int) -> list[SweepRow]:
    candidates = [
        r
        for r in rows
        if r.status == "ok" and r.recall is not None and r.p95_time is not None and float(r.recall) >= recall_threshold
    ]
    candidates.sort(key=lambda r: (float(r.p95_time), -(float(r.rps) if r.rps is not None else 0.0)))
    return candidates[:top_k]


def choose_full_best(rows: list[FullEvalRow], best_metric: str, full_recall_threshold: float) -> list[FullEvalRow]:
    ok = [r for r in rows if r.status == "ok" and r.full_rps is not None and r.full_p95_time is not None]
    if full_recall_threshold > 0:
        ok = [r for r in ok if r.full_recall is not None and float(r.full_recall) >= full_recall_threshold]
    if best_metric == "p95_time":
        ok.sort(
            key=lambda r: (
                float(r.full_p95_time),
                -(float(r.full_recall) if r.full_recall is not None else 0.0),
                -(float(r.full_rps) if r.full_rps is not None else 0.0),
            )
        )
    else:
        ok.sort(
            key=lambda r: (
                -(float(r.full_rps) if r.full_rps is not None else 0.0),
                float(r.full_p95_time),
                -(float(r.full_recall) if r.full_recall is not None else 0.0),
            )
        )
    return ok


def validate_config(cfg: dict[str, Any]) -> None:
    if str(cfg["drift_mode"]) not in {"random", "skewed"}:
        raise ValueError("drift_mode must be random or skewed.")
    if int(cfg["delete_count"]) < 0 or int(cfg["insert_count"]) < 0:
        raise ValueError("delete_count/insert_count must be >= 0.")
    if not (0.0 < float(cfg["sample_ratio"]) <= 1.0):
        raise ValueError("sample_ratio must be in (0,1].")
    if int(cfg["sampled_top_k"]) <= 0:
        raise ValueError("sampled_top_k must be > 0.")
    if int(cfg["final_top_n"]) <= 0:
        raise ValueError("final_top_n must be > 0.")
    if str(cfg["full_best_metric"]) not in {"rps", "p95_time"}:
        raise ValueError("full_best_metric must be rps or p95_time.")


def log_stage_start(name: str) -> float:
    print(f"\n===== [START] {name} =====")
    return time.perf_counter()


def log_stage_end(name: str, start_ts: float) -> float:
    elapsed = time.perf_counter() - start_ts
    print(f"===== [END] {name} | elapsed={elapsed:.2f}s =====\n")
    return elapsed


def main() -> None:
    cfg = deepcopy(CONFIG)
    validate_config(cfg)
    total_start = time.perf_counter()
    stage_timings: list[StageTiming] = []

    run_tag = str(cfg["run_tag"]).strip() or f"drift-adapt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = DEFAULT_NEW_ADAPT_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    config_json_value = str(cfg["config_json"]).strip()
    config_json_path = Path(config_json_value) if config_json_value else DEFAULT_CONFIG_JSON
    if not config_json_path.is_absolute():
        config_json_path = (SCRIPT_DIR / config_json_path).resolve()

    t = log_stage_start("Resolve source dataset")
    source = resolve_source(
        datasets_root=DEFAULT_DATASETS_ROOT,
        source_path=str(cfg["source_path"]),
        distance=str(cfg["distance"]),
        vector_size=int(cfg["vector_size"]),
    )
    if int(cfg["neighbors_top_k"]) <= 0:
        cfg["neighbors_top_k"] = infer_top_k_for_source(source)
    stage_timings.append(StageTiming("resolve_source_dataset", log_stage_end("Resolve source dataset", t)))

    # -------------------- Step 1: Baseline --------------------
    t = log_stage_start("Step 1 Baseline benchmark")
    baseline_dataset = get_dataset_for_benchmark(source, cfg, run_tag)
    baseline_configs = load_json(config_json_path)
    original_engine_configs = deepcopy(baseline_configs)
    engine_cfg = None
    for item in baseline_configs:
        if item.get("name") == str(cfg["engine_name"]):
            engine_cfg = item
            break
    if engine_cfg is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")

    try:
        set_engine_params(
            engine_cfg=engine_cfg,
            index_type=str(cfg["initial_index_type"]),
            index_params=dict(cfg["initial_index_params"]),
            search_params=dict(cfg["initial_search_params"]),
        )
        save_json(config_json_path, baseline_configs)
        baseline_perf = run_benchmark_once(
            engine_name=str(cfg["engine_name"]),
            host=str(cfg["host"]),
            dataset_name=str(baseline_dataset["dataset_name"]),
            dataset_path=str(baseline_dataset["dataset_path"]),
            vector_size=int(baseline_dataset["vector_size"]),
            distance=str(baseline_dataset["distance"]),
            result_meta_path=run_dir / "baseline_result_meta.json",
        )
    finally:
        save_json(config_json_path, original_engine_configs)
    stage_timings.append(StageTiming("step1_baseline_benchmark", log_stage_end("Step 1 Baseline benchmark", t)))

    save_json(
        run_dir / "baseline_performance.json",
        {
            "dataset": baseline_dataset,
            "initial_index_type": cfg["initial_index_type"],
            "initial_index_params": cfg["initial_index_params"],
            "initial_search_params": cfg["initial_search_params"],
            "metrics": asdict(baseline_perf),
        },
    )

    # -------------------- Step 2: Drift simulation --------------------
    t = log_stage_start("Step 2 Drift simulation and dataset build")
    queries = load_queries_from_source(source)
    train_vectors, train_payloads = load_train_vectors_and_payloads(source)
    drifted_vectors, drifted_payloads, drift_meta = simulate_drift(train_vectors, train_payloads, cfg)

    drifted_dir = run_dir / "drifted_full"
    if drifted_dir.exists():
        shutil.rmtree(drifted_dir, ignore_errors=True)
    drifted_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_vectors(drifted_dir / "vectors.jsonl", drifted_vectors)
    write_jsonl_payloads(drifted_dir / "payloads.jsonl", drifted_payloads)
    write_jsonl_queries(drifted_dir / "queries.jsonl", queries)
    actual_top_k = recompute_neighbors(
        vectors=drifted_vectors,
        queries=queries,
        distance=str(source.distance),
        top_k=int(cfg["neighbors_top_k"]),
        out_path=drifted_dir / "neighbours.jsonl",
    )

    drift_info = {
        "source_label": source.label,
        "source_kind": source.kind,
        "source_dir": str(source.source_dir),
        "distance": source.distance,
        "vector_size": int(source.vector_size),
        "neighbors_top_k": int(actual_top_k),
        "drift_mode": cfg["drift_mode"],
        "drift_meta": drift_meta,
        "drifted_dataset_dir": str(drifted_dir),
        "drifted_dataset_name": f"new-adapt-{run_tag}-drifted-full",
    }
    save_json(run_dir / "drift_dataset_info.json", drift_info)
    stage_timings.append(StageTiming("step2_drift_simulation_build_dataset", log_stage_end("Step 2 Drift simulation and dataset build", t)))

    # -------------------- Step 2b: Initial params on drifted full dataset --------------------
    t = log_stage_start("Step 2b Evaluate initial index params on drifted full dataset")
    drifted_initial_configs = load_json(config_json_path)
    drifted_initial_original = deepcopy(drifted_initial_configs)
    drifted_initial_engine = None
    for item in drifted_initial_configs:
        if item.get("name") == str(cfg["engine_name"]):
            drifted_initial_engine = item
            break
    if drifted_initial_engine is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")

    try:
        set_engine_params(
            engine_cfg=drifted_initial_engine,
            index_type=str(cfg["initial_index_type"]),
            index_params=dict(cfg["initial_index_params"]),
            search_params=dict(cfg["initial_search_params"]),
        )
        save_json(config_json_path, drifted_initial_configs)
        drifted_initial_perf = run_benchmark_once(
            engine_name=str(cfg["engine_name"]),
            host=str(cfg["host"]),
            dataset_name=str(drift_info["drifted_dataset_name"]),
            dataset_path=str(drifted_dir),
            vector_size=int(source.vector_size),
            distance=str(source.distance),
            result_meta_path=run_dir / "drifted_initial_result_meta.json",
        )
    finally:
        save_json(config_json_path, drifted_initial_original)
    stage_timings.append(
        StageTiming(
            "step2b_initial_params_on_drifted_dataset",
            log_stage_end("Step 2b Evaluate initial index params on drifted full dataset", t),
        )
    )

    save_json(
        run_dir / "drifted_initial_performance.json",
        {
            "dataset_name": drift_info["drifted_dataset_name"],
            "dataset_path": str(drifted_dir),
            "index_type": cfg["initial_index_type"],
            "index_params": cfg["initial_index_params"],
            "search_params": cfg["initial_search_params"],
            "metrics": asdict(drifted_initial_perf),
        },
    )

    # -------------------- Step 3a: Sampled sweep on drifted data --------------------
    t = log_stage_start("Step 3a Generate sampled dataset on drifted full data")
    sampled_run_tag = f"{run_tag}-sampled-sweep"
    sample_info = run_sampling_generation_for_drifted(
        drifted_dataset_dir=drifted_dir,
        source_distance=str(source.distance),
        source_vector_size=int(source.vector_size),
        cfg=cfg,
        output_name=sampled_run_tag,
    )
    sampled_dataset_name = str(sample_info["sampled_dataset_name"])
    sampled_dataset_path = str(sample_info["sampled_dir"])
    sampled_vector_size = int(sample_info["vector_size"])
    sampled_distance = str(sample_info["distance"])
    stage_timings.append(StageTiming("step3a_generate_sampled_dataset", log_stage_end("Step 3a Generate sampled dataset on drifted full data", t)))

    t = log_stage_start("Step 3a Run sampled parameter sweep")
    index_grid = normalize_grid(dict(cfg["sweep_index_param_grid"]), "sweep_index_param_grid")
    search_grid = normalize_grid(dict(cfg["sweep_search_param_grid"]), "sweep_search_param_grid")
    index_combos = cartesian_product(index_grid)
    search_combos = cartesian_product(search_grid)

    sweep_rows: list[SweepRow] = []
    sweep_configs = load_json(config_json_path)
    sweep_original = deepcopy(sweep_configs)
    sweep_engine = None
    for item in sweep_configs:
        if item.get("name") == str(cfg["engine_name"]):
            sweep_engine = item
            break
    if sweep_engine is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")

    try:
        seq = 0
        total = len(index_combos) * len(search_combos)
        for idx_params in index_combos:
            for s_params in search_combos:
                seq += 1
                print(f"[sampled sweep {seq}/{total}] index={idx_params}, search={s_params}")
                set_engine_params(
                    engine_cfg=sweep_engine,
                    index_type=str(cfg["sweep_index_type"]),
                    index_params=idx_params,
                    search_params=s_params,
                )
                save_json(config_json_path, sweep_configs)
                try:
                    perf = run_benchmark_once(
                        engine_name=str(cfg["engine_name"]),
                        host=str(cfg["host"]),
                        dataset_name=sampled_dataset_name,
                        dataset_path=sampled_dataset_path,
                        vector_size=sampled_vector_size,
                        distance=sampled_distance,
                        result_meta_path=run_dir / "sampled_sweep_results" / f"combo-{seq:04d}.json",
                    )
                    sweep_rows.append(
                        SweepRow(
                            combo_id=seq,
                            index_type=str(cfg["sweep_index_type"]),
                            index_params=dict(idx_params),
                            search_params=dict(s_params),
                            rps=perf.rps,
                            p95_time=perf.p95_time,
                            recall=perf.recall,
                            result_file=perf.result_file,
                            status="ok",
                            error="",
                        )
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    sweep_rows.append(
                        SweepRow(
                            combo_id=seq,
                            index_type=str(cfg["sweep_index_type"]),
                            index_params=dict(idx_params),
                            search_params=dict(s_params),
                            rps=None,
                            p95_time=None,
                            recall=None,
                            result_file="",
                            status="failed",
                            error=str(exc),
                        )
                    )
                    if not bool(cfg["continue_on_sweep_error"]):
                        raise
    finally:
        save_json(config_json_path, sweep_original)

    save_json(run_dir / "drift_sampled_sweep_results.json", [asdict(r) for r in sweep_rows])

    sampled_topk = choose_sampled_topk(
        rows=sweep_rows,
        recall_threshold=float(cfg["sampled_recall_threshold"]),
        top_k=int(cfg["sampled_top_k"]),
    )
    if not sampled_topk:
        raise RuntimeError(
            "No sampled sweep candidate satisfies sampled_recall_threshold. "
            "Adjust threshold or sweep params."
        )
    save_json(run_dir / "drift_sampled_topk_candidates.json", [asdict(r) for r in sampled_topk])
    stage_timings.append(StageTiming("step3a_sampled_param_sweep", log_stage_end("Step 3a Run sampled parameter sweep", t)))

    # -------------------- Step 3b: Full eval on drifted full data --------------------
    t = log_stage_start("Step 3b Full evaluation on drifted full dataset")
    full_rows: list[FullEvalRow] = []
    full_configs = load_json(config_json_path)
    full_original = deepcopy(full_configs)
    full_engine = None
    for item in full_configs:
        if item.get("name") == str(cfg["engine_name"]):
            full_engine = item
            break
    if full_engine is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")

    drifted_dataset_name = str(drift_info["drifted_dataset_name"])
    try:
        for rank, row in enumerate(sampled_topk, start=1):
            print(
                f"[full eval {rank}/{len(sampled_topk)}] combo_id={row.combo_id}, "
                f"index={row.index_params}, search={row.search_params}"
            )
            set_engine_params(
                engine_cfg=full_engine,
                index_type=row.index_type,
                index_params=row.index_params,
                search_params=row.search_params,
            )
            save_json(config_json_path, full_configs)
            try:
                perf = run_benchmark_once(
                    engine_name=str(cfg["engine_name"]),
                    host=str(cfg["host"]),
                    dataset_name=drifted_dataset_name,
                    dataset_path=str(drifted_dir),
                    vector_size=int(source.vector_size),
                    distance=str(source.distance),
                    result_meta_path=run_dir / "drift_full_eval_results" / f"rank-{rank:02d}.json",
                )
                full_rows.append(
                    FullEvalRow(
                        sampled_rank=rank,
                        sampled_combo_id=row.combo_id,
                        index_type=row.index_type,
                        index_params=row.index_params,
                        search_params=row.search_params,
                        sampled_rps=row.rps,
                        sampled_p95_time=row.p95_time,
                        sampled_recall=row.recall,
                        full_rps=perf.rps,
                        full_p95_time=perf.p95_time,
                        full_recall=perf.recall,
                        full_result_file=perf.result_file,
                        status="ok",
                        error="",
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                full_rows.append(
                    FullEvalRow(
                        sampled_rank=rank,
                        sampled_combo_id=row.combo_id,
                        index_type=row.index_type,
                        index_params=row.index_params,
                        search_params=row.search_params,
                        sampled_rps=row.rps,
                        sampled_p95_time=row.p95_time,
                        sampled_recall=row.recall,
                        full_rps=None,
                        full_p95_time=None,
                        full_recall=None,
                        full_result_file="",
                        status="failed",
                        error=str(exc),
                    )
                )
    finally:
        save_json(config_json_path, full_original)

    save_json(run_dir / "drift_full_topk_eval_results.json", [asdict(r) for r in full_rows])
    stage_timings.append(StageTiming("step3b_full_eval_on_drifted_dataset", log_stage_end("Step 3b Full evaluation on drifted full dataset", t)))

    t = log_stage_start("Step 4 Final ranking and summary generation")
    ranked_full = choose_full_best(
        rows=full_rows,
        best_metric=str(cfg["full_best_metric"]),
        full_recall_threshold=float(cfg["full_recall_threshold"]),
    )
    final_top_n = min(int(cfg["final_top_n"]), len(ranked_full))
    final_top = ranked_full[:final_top_n]

    top3_with_compare: list[dict[str, Any]] = []
    for i, item in enumerate(final_top, start=1):
        top3_with_compare.append(
            {
                "rank": i,
                "config": {
                    "index_type": item.index_type,
                    "index_params": item.index_params,
                    "search_params": item.search_params,
                },
                "sampled_metrics": {
                    "rps": item.sampled_rps,
                    "p95_time": item.sampled_p95_time,
                    "recall": item.sampled_recall,
                },
                "drifted_full_metrics": {
                    "rps": item.full_rps,
                    "p95_time": item.full_p95_time,
                    "recall": item.full_recall,
                },
                "vs_baseline": {
                    "delta_rps": (None if (item.full_rps is None or baseline_perf.rps is None) else item.full_rps - baseline_perf.rps),
                    "delta_p95_time": (
                        None
                        if (item.full_p95_time is None or baseline_perf.p95_time is None)
                        else item.full_p95_time - baseline_perf.p95_time
                    ),
                    "delta_recall": (
                        None
                        if (item.full_recall is None or baseline_perf.recall is None)
                        else item.full_recall - baseline_perf.recall
                    ),
                },
                "vs_drifted_initial": {
                    "delta_rps": (
                        None
                        if (item.full_rps is None or drifted_initial_perf.rps is None)
                        else item.full_rps - drifted_initial_perf.rps
                    ),
                    "delta_p95_time": (
                        None
                        if (item.full_p95_time is None or drifted_initial_perf.p95_time is None)
                        else item.full_p95_time - drifted_initial_perf.p95_time
                    ),
                    "delta_recall": (
                        None
                        if (item.full_recall is None or drifted_initial_perf.recall is None)
                        else item.full_recall - drifted_initial_perf.recall
                    ),
                },
                "result_file": item.full_result_file,
            }
        )

    final_summary = {
        "run_tag": run_tag,
        "source_dataset": {
            "label": source.label,
            "kind": source.kind,
            "source_dir": str(source.source_dir),
            "distance": source.distance,
            "vector_size": int(source.vector_size),
        },
        "baseline": {
            "index_type": cfg["initial_index_type"],
            "index_params": cfg["initial_index_params"],
            "search_params": cfg["initial_search_params"],
            "metrics": asdict(baseline_perf),
        },
        "drift": drift_info,
        "drifted_initial": {
            "index_type": cfg["initial_index_type"],
            "index_params": cfg["initial_index_params"],
            "search_params": cfg["initial_search_params"],
            "metrics": asdict(drifted_initial_perf),
        },
        "sampled_selection": {
            "sampled_recall_threshold": float(cfg["sampled_recall_threshold"]),
            "sampled_top_k": int(cfg["sampled_top_k"]),
            "selected_count": len(sampled_topk),
        },
        "full_eval": {
            "best_metric": cfg["full_best_metric"],
            "full_recall_threshold": float(cfg["full_recall_threshold"]),
            "evaluated_count": len(full_rows),
            "ok_count": sum(1 for r in full_rows if r.status == "ok"),
            "failed_count": sum(1 for r in full_rows if r.status != "ok"),
        },
        "final_top3": top3_with_compare,
        "artifacts": {
            "drift_dataset_info": str(run_dir / "drift_dataset_info.json"),
            "baseline_performance": str(run_dir / "baseline_performance.json"),
            "drifted_initial_performance": str(run_dir / "drifted_initial_performance.json"),
            "sampled_sweep_results": str(run_dir / "drift_sampled_sweep_results.json"),
            "sampled_topk": str(run_dir / "drift_sampled_topk_candidates.json"),
            "full_eval_results": str(run_dir / "drift_full_topk_eval_results.json"),
        },
    }
    stage_timings.append(StageTiming("step4_final_summary_generation", log_stage_end("Step 4 Final ranking and summary generation", t)))
    total_elapsed = time.perf_counter() - total_start
    final_summary["timing"] = {
        "stages": [asdict(x) for x in stage_timings],
        "total_seconds": total_elapsed,
    }
    save_json(run_dir / "drift_adaptive_top3_summary.json", final_summary)

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print("===== [TIMING] Stage breakdown =====")
    for item in stage_timings:
        print(f"- {item.name}: {item.seconds:.2f}s")
    print(f"===== [TIMING] Total elapsed: {total_elapsed:.2f}s =====")
    print(f"Saved summary: {run_dir / 'drift_adaptive_top3_summary.json'}")


if __name__ == "__main__":
    main()
