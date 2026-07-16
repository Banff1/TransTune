#!/usr/bin/env python3
"""Drift adaptive tuning pipeline with Bayesian optimization on sampled data.

This script keeps the same end-to-end workflow as run_drift_adaptive_pipeline.py,
but replaces Step 3a (exhaustive grid sweep on sampled dataset) with
Bayesian optimization over the same discrete candidate space.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Limit BLAS/OpenMP threads BEFORE importing numpy/scikit-learn.
# In nohup/background runs, this avoids OpenBLAS thread-metadata overflow.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
os.environ.setdefault("GOTO_NUM_THREADS", "16")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "16")
os.environ.setdefault("BLIS_NUM_THREADS", "16")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import run_drift_adaptive_pipeline as base


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
    "initial_index_params": {"nlist": 101},
    "initial_search_params": {"nprobe": 25, "reorder_k": 160},

    # Drift simulation
    # drift_mode: "random" | "skewed"
    "drift_mode": "skewed",
    "delete_count": 20000,
    "insert_count": 600000,
    "insert_noise_std": 0.01,
    # skewed mode params
    "skewed_n_clusters": 20,
    "skewed_m_hot_clusters": 1,
    "skewed_hot_fraction": 0.9,  # fraction of insert/delete applied to hot clusters
    "skewed_cold_fraction": 0.1,  # remaining clusters
    "skewed_kmeans_batch_size": 4096,

    # Sampled search on drifted dataset
    "sample_ratio": 0.1,
    "sampling_strategy": "kmeans_stratified",  # uniform | kmeans_stratified
    "sampling_kmeans_clusters": 50,
    "sampling_kmeans_batch_size": 2048,
    "bo_index_type": "SCANN",
    # BO parameter space: support either explicit list or numeric range dict.
    # Range format: {"start": a, "stop": b, "step": c} where stop is inclusive.
    "bo_index_param_space": {
        "nlist": {"start": 100, "stop": 500, "step": 50},
    },
    "bo_search_param_space": {
        "nprobe": {"start": 50, "stop": 500, "step": 25},
        "reorder_k": [150, 200, 225, 250, 275, 300, 325, 350, 375, 400],
    },
    # "bo_index_param_space": {
    #     "nlist": {"start": 500, "stop": 1500, "step": 50},
    # },
    # "bo_search_param_space": {
    #     "nprobe": {"start": 100, "stop": 500, "step": 10},
    #     "reorder_k": [400, 425, 450, 475, 500, 525, 550, 575, 600],
    # },    
    "sampled_recall_threshold": 0.9,
    "sampled_top_k": 5,
    "continue_on_sweep_error": True,

    # Full drifted dataset evaluation
    "full_best_metric": "rps",  # "rps" | "p95_time"
    "full_recall_threshold": 0.85,  # <=0 means disabled

    # Final output count
    "final_top_n": 3,

    # Optional run tag
    "run_tag": "",  # empty => auto generated

    # Step 3a Bayesian optimization budget
    "sampled_bo_init_points": 10,
    "sampled_bo_iterations": 20,
    # Exploration strength in EI
    "sampled_bo_xi": 0.01,
    # Optional cap for max evaluations in Step 3a
    # <=0 means use init_points + iterations
    "sampled_bo_max_evals": 0,
}


def validate_bo_config(cfg: dict[str, Any]) -> None:
    if int(cfg["sampled_bo_init_points"]) <= 0:
        raise ValueError("sampled_bo_init_points must be > 0.")
    if int(cfg["sampled_bo_iterations"]) < 0:
        raise ValueError("sampled_bo_iterations must be >= 0.")
    if float(cfg["sampled_bo_xi"]) < 0:
        raise ValueError("sampled_bo_xi must be >= 0.")
    if not str(cfg["bo_index_type"]).strip():
        raise ValueError("bo_index_type cannot be empty.")


def expand_numeric_range(spec: dict[str, Any], field_name: str) -> list[int | float]:
    if "start" not in spec or "stop" not in spec or "step" not in spec:
        raise ValueError(f"{field_name} range must include start/stop/step.")
    start = spec["start"]
    stop = spec["stop"]
    step = spec["step"]
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)) or not isinstance(step, (int, float)):
        raise ValueError(f"{field_name} range start/stop/step must be numeric.")
    if float(step) == 0.0:
        raise ValueError(f"{field_name} range step cannot be 0.")

    values: list[int | float] = []
    use_int = isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)
    if float(step) > 0 and float(start) > float(stop):
        raise ValueError(f"{field_name} range has positive step but start > stop.")
    if float(step) < 0 and float(start) < float(stop):
        raise ValueError(f"{field_name} range has negative step but start < stop.")

    current = float(start)
    stop_f = float(stop)
    step_f = float(step)
    eps = 1e-12
    if step_f > 0:
        while current <= stop_f + eps:
            values.append(int(round(current)) if use_int else current)
            current += step_f
    else:
        while current >= stop_f - eps:
            values.append(int(round(current)) if use_int else current)
            current += step_f
    if not values:
        raise ValueError(f"{field_name} range expands to empty values.")
    return values


def normalize_param_space(space: dict[str, Any], field_name: str) -> dict[str, list[Any]]:
    if not isinstance(space, dict):
        raise ValueError(f"{field_name} must be dict.")

    normalized: dict[str, list[Any]] = {}
    for key, raw in space.items():
        key_name = str(key)
        item_name = f"{field_name}.{key_name}"
        if isinstance(raw, dict) and any(k in raw for k in ("start", "stop", "step")):
            values = expand_numeric_range(raw, item_name)
        elif isinstance(raw, dict) and "values" in raw:
            vals = raw["values"]
            if not isinstance(vals, list):
                raise ValueError(f"{item_name}.values must be a list.")
            values = vals
        elif isinstance(raw, list):
            values = raw
        else:
            values = [raw]

        if not values:
            raise ValueError(f"{item_name} cannot be empty.")
        normalized[key_name] = values

    return normalized


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    eps = 1e-12
    std_safe = np.maximum(std, eps)
    improvement = mean - best_y - xi
    z = improvement / std_safe
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    ei = improvement * cdf + std_safe * pdf
    ei = np.where(std > eps, ei, 0.0)
    return np.asarray(ei, dtype=float)


def utility_from_metrics(
    rps: float | None,
    p95_time: float | None,
    recall: float | None,
    recall_threshold: float,
) -> float:
    if rps is None or p95_time is None or recall is None:
        return -1e9

    recall_v = float(recall)
    p95_v = max(float(p95_time), 1e-9)
    rps_v = max(float(rps), 0.0)

    if recall_v < recall_threshold:
        # Hard penalty to prioritize feasible (recall-qualified) region first.
        return -1e6 - (recall_threshold - recall_v) * 1e5 - p95_v

    return recall_v * 1000.0 + (1000.0 / p95_v) + math.log1p(rps_v)


def build_discrete_candidates(
    index_grid: dict[str, list[Any]],
    search_grid: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    index_combos = base.cartesian_product(index_grid)
    search_combos = base.cartesian_product(search_grid)

    # Keep deterministic order to make run reproducible.
    all_candidates: list[dict[str, Any]] = []
    for idx_params in index_combos:
        for search_params in search_combos:
            all_candidates.append(
                {
                    "index_params": dict(idx_params),
                    "search_params": dict(search_params),
                }
            )

    # Encode each discrete parameter as ordinal index in [0, 1].
    param_values: dict[str, list[Any]] = {}
    for grid in (index_grid, search_grid):
        for k, vals in grid.items():
            uniq = []
            for x in vals:
                if x not in uniq:
                    uniq.append(x)
            param_values[k] = uniq

    feat_keys = sorted(param_values.keys())
    features: list[np.ndarray] = []
    for candidate in all_candidates:
        feature_vals: list[float] = []
        merged = dict(candidate["index_params"])
        merged.update(candidate["search_params"])
        for key in feat_keys:
            vals = param_values[key]
            idx = vals.index(merged[key])
            denom = max(len(vals) - 1, 1)
            feature_vals.append(float(idx) / float(denom))
        features.append(np.asarray(feature_vals, dtype=np.float64))

    return all_candidates, features


def run_sampled_bo_search(
    cfg: dict[str, Any],
    run_dir: Path,
    config_json_path: Path,
    sampled_dataset_name: str,
    sampled_dataset_path: str,
    sampled_vector_size: int,
    sampled_distance: str,
) -> list[base.SweepRow]:
    index_grid = normalize_param_space(dict(cfg["bo_index_param_space"]), "bo_index_param_space")
    search_grid = normalize_param_space(dict(cfg["bo_search_param_space"]), "bo_search_param_space")
    candidates, feature_list = build_discrete_candidates(index_grid, search_grid)
    total_candidates = len(candidates)
    if total_candidates == 0:
        raise RuntimeError("No candidate combinations generated for BO search.")

    cfgs = base.load_json(config_json_path)
    cfgs_original = deepcopy(cfgs)
    engine_cfg = None
    for item in cfgs:
        if item.get("name") == str(cfg["engine_name"]):
            engine_cfg = item
            break
    if engine_cfg is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")

    rng = random.Random(int(cfg["seed"]))
    init_points = min(int(cfg["sampled_bo_init_points"]), total_candidates)
    default_budget = init_points + int(cfg["sampled_bo_iterations"])
    max_evals_cfg = int(cfg["sampled_bo_max_evals"])
    max_evals = default_budget if max_evals_cfg <= 0 else min(max_evals_cfg, total_candidates)

    all_ids = list(range(total_candidates))
    rng.shuffle(all_ids)
    init_queue: list[int] = all_ids[:init_points]
    remaining_ids: set[int] = set(all_ids[init_points:])

    rows: list[base.SweepRow] = []
    y_values: list[float] = []
    x_values: list[np.ndarray] = []
    combo_seq = 0

    try:
        while len(rows) < max_evals and (init_queue or remaining_ids):
            next_id: int | None = None
            if init_queue:
                # Consume random initial design first.
                next_id = init_queue.pop()
            else:
                if not remaining_ids:
                    break
                if len(x_values) < 3:
                    next_id = rng.choice(list(remaining_ids))
                else:
                    x_train = np.vstack(x_values)
                    y_train = np.asarray(y_values, dtype=np.float64)
                    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                        length_scale=np.ones(x_train.shape[1], dtype=np.float64),
                        length_scale_bounds=(1e-3, 1e3),
                        nu=2.5,
                    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1))
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-8,
                        normalize_y=True,
                        random_state=int(cfg["seed"]),
                        n_restarts_optimizer=2,
                    )
                    gp.fit(x_train, y_train)
                    rem = sorted(remaining_ids)
                    rem_x = np.vstack([feature_list[i] for i in rem])
                    mean, std = gp.predict(rem_x, return_std=True)
                    ei = expected_improvement(
                        mean=np.asarray(mean, dtype=np.float64),
                        std=np.asarray(std, dtype=np.float64),
                        best_y=float(np.max(y_train)),
                        xi=float(cfg["sampled_bo_xi"]),
                    )
                    next_id = rem[int(np.argmax(ei))]

            if next_id is None:
                break
            if next_id not in remaining_ids and not init_queue:
                continue
            remaining_ids.discard(next_id)

            combo_seq += 1
            candidate = candidates[next_id]
            index_params = dict(candidate["index_params"])
            search_params = dict(candidate["search_params"])
            print(
                f"[sampled BO {combo_seq}/{max_evals}] candidate_id={next_id + 1}/{total_candidates} "
                f"index={index_params}, search={search_params}"
            )
            base.set_engine_params(
                engine_cfg=engine_cfg,
                index_type=str(cfg["bo_index_type"]),
                index_params=index_params,
                search_params=search_params,
            )
            base.save_json(config_json_path, cfgs)
            try:
                perf = base.run_benchmark_once(
                    engine_name=str(cfg["engine_name"]),
                    host=str(cfg["host"]),
                    dataset_name=sampled_dataset_name,
                    dataset_path=sampled_dataset_path,
                    vector_size=sampled_vector_size,
                    distance=sampled_distance,
                    result_meta_path=run_dir / "sampled_bo_results" / f"bo-{combo_seq:04d}.json",
                )
                rows.append(
                    base.SweepRow(
                        combo_id=next_id + 1,
                        index_type=str(cfg["bo_index_type"]),
                        index_params=index_params,
                        search_params=search_params,
                        rps=perf.rps,
                        p95_time=perf.p95_time,
                        recall=perf.recall,
                        result_file=perf.result_file,
                        status="ok",
                        error="",
                    )
                )
                x_values.append(feature_list[next_id])
                y_values.append(
                    utility_from_metrics(
                        rps=perf.rps,
                        p95_time=perf.p95_time,
                        recall=perf.recall,
                        recall_threshold=float(cfg["sampled_recall_threshold"]),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                rows.append(
                    base.SweepRow(
                        combo_id=next_id + 1,
                        index_type=str(cfg["bo_index_type"]),
                        index_params=index_params,
                        search_params=search_params,
                        rps=None,
                        p95_time=None,
                        recall=None,
                        result_file="",
                        status="failed",
                        error=str(exc),
                    )
                )
                x_values.append(feature_list[next_id])
                y_values.append(-1e9)
                if not bool(cfg["continue_on_sweep_error"]):
                    raise

            if not remaining_ids:
                break
            if len(rows) >= max_evals:
                break
    finally:
        base.save_json(config_json_path, cfgs_original)

    return rows


def main() -> None:
    cfg = deepcopy(CONFIG)
    base.validate_config(cfg)
    validate_bo_config(cfg)

    total_start = time.perf_counter()
    stage_timings: list[base.StageTiming] = []
    run_tag = str(cfg["run_tag"]).strip() or f"drift-adapt-bo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = base.DEFAULT_NEW_ADAPT_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    config_json_value = str(cfg["config_json"]).strip()
    config_json_path = Path(config_json_value) if config_json_value else base.DEFAULT_CONFIG_JSON
    if not config_json_path.is_absolute():
        config_json_path = (base.SCRIPT_DIR / config_json_path).resolve()

    t = base.log_stage_start("Resolve source dataset")
    source = base.resolve_source(
        datasets_root=base.DEFAULT_DATASETS_ROOT,
        source_path=str(cfg["source_path"]),
        distance=str(cfg["distance"]),
        vector_size=int(cfg["vector_size"]),
    )
    if int(cfg["neighbors_top_k"]) <= 0:
        cfg["neighbors_top_k"] = base.infer_top_k_for_source(source)
    stage_timings.append(base.StageTiming("resolve_source_dataset", base.log_stage_end("Resolve source dataset", t)))

    # Step 1
    t = base.log_stage_start("Step 1 Baseline benchmark")
    baseline_dataset = base.get_dataset_for_benchmark(source, cfg, run_tag)
    baseline_configs = base.load_json(config_json_path)
    baseline_original = deepcopy(baseline_configs)
    baseline_engine = None
    for item in baseline_configs:
        if item.get("name") == str(cfg["engine_name"]):
            baseline_engine = item
            break
    if baseline_engine is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")
    try:
        base.set_engine_params(
            engine_cfg=baseline_engine,
            index_type=str(cfg["initial_index_type"]),
            index_params=dict(cfg["initial_index_params"]),
            search_params=dict(cfg["initial_search_params"]),
        )
        base.save_json(config_json_path, baseline_configs)
        baseline_perf = base.run_benchmark_once(
            engine_name=str(cfg["engine_name"]),
            host=str(cfg["host"]),
            dataset_name=str(baseline_dataset["dataset_name"]),
            dataset_path=str(baseline_dataset["dataset_path"]),
            vector_size=int(baseline_dataset["vector_size"]),
            distance=str(baseline_dataset["distance"]),
            result_meta_path=run_dir / "baseline_result_meta.json",
        )
    finally:
        base.save_json(config_json_path, baseline_original)
    stage_timings.append(base.StageTiming("step1_baseline_benchmark", base.log_stage_end("Step 1 Baseline benchmark", t)))

    base.save_json(
        run_dir / "baseline_performance.json",
        {
            "dataset": baseline_dataset,
            "initial_index_type": cfg["initial_index_type"],
            "initial_index_params": cfg["initial_index_params"],
            "initial_search_params": cfg["initial_search_params"],
            "metrics": asdict(baseline_perf),
        },
    )

    # Step 2
    t = base.log_stage_start("Step 2 Drift simulation and dataset build")
    queries = base.load_queries_from_source(source)
    train_vectors, train_payloads = base.load_train_vectors_and_payloads(source)
    drifted_vectors, drifted_payloads, drift_meta = base.simulate_drift(train_vectors, train_payloads, cfg)

    drifted_dir = run_dir / "drifted_full"
    if drifted_dir.exists():
        import shutil

        shutil.rmtree(drifted_dir, ignore_errors=True)
    drifted_dir.mkdir(parents=True, exist_ok=True)
    base.write_jsonl_vectors(drifted_dir / "vectors.jsonl", drifted_vectors)
    base.write_jsonl_payloads(drifted_dir / "payloads.jsonl", drifted_payloads)
    base.write_jsonl_queries(drifted_dir / "queries.jsonl", queries)
    actual_top_k = base.recompute_neighbors(
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
    base.save_json(run_dir / "drift_dataset_info.json", drift_info)
    stage_timings.append(
        base.StageTiming("step2_drift_simulation_build_dataset", base.log_stage_end("Step 2 Drift simulation and dataset build", t))
    )

    # Step 2b
    t = base.log_stage_start("Step 2b Evaluate initial index params on drifted full dataset")
    drift_cfgs = base.load_json(config_json_path)
    drift_orig = deepcopy(drift_cfgs)
    drift_engine = None
    for item in drift_cfgs:
        if item.get("name") == str(cfg["engine_name"]):
            drift_engine = item
            break
    if drift_engine is None:
        raise KeyError(f"Engine {cfg['engine_name']} not found in {config_json_path}")
    try:
        base.set_engine_params(
            engine_cfg=drift_engine,
            index_type=str(cfg["initial_index_type"]),
            index_params=dict(cfg["initial_index_params"]),
            search_params=dict(cfg["initial_search_params"]),
        )
        base.save_json(config_json_path, drift_cfgs)
        drifted_initial_perf = base.run_benchmark_once(
            engine_name=str(cfg["engine_name"]),
            host=str(cfg["host"]),
            dataset_name=str(drift_info["drifted_dataset_name"]),
            dataset_path=str(drifted_dir),
            vector_size=int(source.vector_size),
            distance=str(source.distance),
            result_meta_path=run_dir / "drifted_initial_result_meta.json",
        )
    finally:
        base.save_json(config_json_path, drift_orig)
    stage_timings.append(
        base.StageTiming(
            "step2b_initial_params_on_drifted_dataset",
            base.log_stage_end("Step 2b Evaluate initial index params on drifted full dataset", t),
        )
    )

    base.save_json(
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

    # Step 3a sampled + BO
    t = base.log_stage_start("Step 3a Generate sampled dataset on drifted full data")
    sampled_run_tag = f"{run_tag}-sampled-bo"
    sample_info = base.run_sampling_generation_for_drifted(
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
    stage_timings.append(
        base.StageTiming(
            "step3a_generate_sampled_dataset",
            base.log_stage_end("Step 3a Generate sampled dataset on drifted full data", t),
        )
    )

    t = base.log_stage_start("Step 3a Run sampled Bayesian optimization")
    sweep_rows = run_sampled_bo_search(
        cfg=cfg,
        run_dir=run_dir,
        config_json_path=config_json_path,
        sampled_dataset_name=sampled_dataset_name,
        sampled_dataset_path=sampled_dataset_path,
        sampled_vector_size=sampled_vector_size,
        sampled_distance=sampled_distance,
    )
    base.save_json(run_dir / "drift_sampled_bo_results.json", [asdict(r) for r in sweep_rows])
    sampled_topk = base.choose_sampled_topk(
        rows=sweep_rows,
        recall_threshold=float(cfg["sampled_recall_threshold"]),
        top_k=int(cfg["sampled_top_k"]),
    )
    if not sampled_topk:
        raise RuntimeError("No BO candidate satisfies sampled_recall_threshold. Please relax threshold or increase BO budget.")
    base.save_json(run_dir / "drift_sampled_topk_candidates.json", [asdict(r) for r in sampled_topk])
    stage_timings.append(
        base.StageTiming("step3a_sampled_param_bo", base.log_stage_end("Step 3a Run sampled Bayesian optimization", t))
    )

    # Step 3b
    t = base.log_stage_start("Step 3b Full evaluation on drifted full dataset")
    full_rows: list[base.FullEvalRow] = []
    full_cfgs = base.load_json(config_json_path)
    full_original = deepcopy(full_cfgs)
    full_engine = None
    for item in full_cfgs:
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
            base.set_engine_params(
                engine_cfg=full_engine,
                index_type=row.index_type,
                index_params=row.index_params,
                search_params=row.search_params,
            )
            base.save_json(config_json_path, full_cfgs)
            try:
                perf = base.run_benchmark_once(
                    engine_name=str(cfg["engine_name"]),
                    host=str(cfg["host"]),
                    dataset_name=drifted_dataset_name,
                    dataset_path=str(drifted_dir),
                    vector_size=int(source.vector_size),
                    distance=str(source.distance),
                    result_meta_path=run_dir / "drift_full_eval_results" / f"rank-{rank:02d}.json",
                )
                full_rows.append(
                    base.FullEvalRow(
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
                    base.FullEvalRow(
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
        base.save_json(config_json_path, full_original)
    base.save_json(run_dir / "drift_full_topk_eval_results.json", [asdict(r) for r in full_rows])
    stage_timings.append(
        base.StageTiming("step3b_full_eval_on_drifted_dataset", base.log_stage_end("Step 3b Full evaluation on drifted full dataset", t))
    )

    # Step 4 summary
    t = base.log_stage_start("Step 4 Final ranking and summary generation")
    ranked_full = base.choose_full_best(
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
                    "delta_rps": None if (item.full_rps is None or baseline_perf.rps is None) else item.full_rps - baseline_perf.rps,
                    "delta_p95_time": (
                        None
                        if (item.full_p95_time is None or baseline_perf.p95_time is None)
                        else item.full_p95_time - baseline_perf.p95_time
                    ),
                    "delta_recall": None if (item.full_recall is None or baseline_perf.recall is None) else item.full_recall - baseline_perf.recall,
                },
                "vs_drifted_initial": {
                    "delta_rps": (
                        None if (item.full_rps is None or drifted_initial_perf.rps is None) else item.full_rps - drifted_initial_perf.rps
                    ),
                    "delta_p95_time": (
                        None
                        if (item.full_p95_time is None or drifted_initial_perf.p95_time is None)
                        else item.full_p95_time - drifted_initial_perf.p95_time
                    ),
                    "delta_recall": (
                        None if (item.full_recall is None or drifted_initial_perf.recall is None) else item.full_recall - drifted_initial_perf.recall
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
            "optimizer": "bayesian_optimization_gp_ei",
            "sampled_recall_threshold": float(cfg["sampled_recall_threshold"]),
            "sampled_top_k": int(cfg["sampled_top_k"]),
            "selected_count": len(sampled_topk),
            "bo_budget": {
                "init_points": int(cfg["sampled_bo_init_points"]),
                "iterations": int(cfg["sampled_bo_iterations"]),
                "max_evals": int(cfg["sampled_bo_max_evals"]),
                "xi": float(cfg["sampled_bo_xi"]),
            },
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
            "sampled_bo_results": str(run_dir / "drift_sampled_bo_results.json"),
            "sampled_topk": str(run_dir / "drift_sampled_topk_candidates.json"),
            "full_eval_results": str(run_dir / "drift_full_topk_eval_results.json"),
        },
    }
    stage_timings.append(base.StageTiming("step4_final_summary_generation", base.log_stage_end("Step 4 Final ranking and summary generation", t)))

    total_elapsed = time.perf_counter() - total_start
    final_summary["timing"] = {"stages": [asdict(x) for x in stage_timings], "total_seconds": total_elapsed}
    base.save_json(run_dir / "drift_adaptive_top3_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print("===== [TIMING] Stage breakdown =====")
    for item in stage_timings:
        print(f"- {item.name}: {item.seconds:.2f}s")
    print(f"===== [TIMING] Total elapsed: {total_elapsed:.2f}s =====")
    print(f"Saved summary: {run_dir / 'drift_adaptive_top3_summary.json'}")


if __name__ == "__main__":
    main()
