#!/usr/bin/env python3
"""Sample dataset, sweep index params, and rank by recall + p95 latency.

Usage:
1) Edit CONFIG below.
2) Run: python3 auto-configure/vdtuner/new_adapt/run_sampling_index_sweep.py
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ===================== User Config =====================
# Put all runtime parameters here instead of CLI arguments.
CONFIG: dict[str, Any] = {
    # Dataset sampling settings
    "source_path": "glove-100-angular",
    "sample_ratio": 0.05,
    "sampling_strategy": "kmeans_stratified",  # uniform | kmeans_stratified
    "kmeans_clusters": 50,
    "kmeans_batch_size": 2048,
    "seed": 42,
    "distance": "",  # empty means auto-resolve from dataset config
    "vector_size": 0,  # 0 means auto-detect
    "neighbors_top_k": 0,  # 0 means infer from source or fallback

    # Benchmark engine settings
    "engine_name": "milvus-p10",
    "host": "127.0.0.1",
    "config_json": "",  # empty means default milvus-single-node config

    # Index/search sweep settings
    "index_type": "SCANN",
    "index_param_grid": {"nlist": [100, 200, 300]},
    "search_param_grid": {"nprobe": [10, 30, 50], "reorder_k": [100, 150]},

    # Result filtering
    "recall_threshold": 0.90,
    "top_n": 5,
    "continue_on_error": True,
    "run_tag": "",  # empty means auto-generated
}
# =======================================================


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "vector-db-benchmark-master"
DEFAULT_CONFIG_JSON = (
    BENCHMARK_ROOT
    / "experiments"
    / "configurations"
    / "milvus-single-node.json"
)


@dataclass
class SweepRecord:
    combo_id: int
    run_tag: str
    index_type: str
    index_params: dict[str, Any]
    search_params: dict[str, Any]
    rps: float | None
    p95_time: float | None
    recall: float | None
    result_file: str
    status: str
    error: str


def cartesian_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for items in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, items)})
    return combos


def normalize_grid(grid: dict[str, Any], field_name: str) -> dict[str, list[Any]]:
    if not isinstance(grid, dict):
        raise ValueError(f"{field_name} must be dict.")
    out: dict[str, list[Any]] = {}
    for k, v in grid.items():
        values = v if isinstance(v, list) else [v]
        if not values:
            raise ValueError(f"{field_name}.{k} candidate list cannot be empty.")
        out[str(k)] = values
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(config_path: Path) -> list[dict[str, Any]]:
    data = load_json(config_path)
    if not isinstance(data, list):
        raise ValueError(f"Config must be JSON list: {config_path}")
    return data


def save_config(config_path: Path, configs: list[dict[str, Any]]) -> None:
    save_json(config_path, configs)


def find_engine(configs: list[dict[str, Any]], engine_name: str) -> dict[str, Any]:
    for cfg in configs:
        if cfg.get("name") == engine_name:
            return cfg
    raise KeyError(f"Engine '{engine_name}' not found in config.")


def set_engine_params(
    engine_cfg: dict[str, Any],
    index_type: str,
    index_params: dict[str, Any],
    search_params: dict[str, Any],
) -> None:
    upload_params = engine_cfg.setdefault("upload_params", {})
    upload_params["index_type"] = index_type
    upload_params["index_params"] = dict(index_params)

    search_cfg = engine_cfg.get("search_params")
    if not isinstance(search_cfg, list) or not search_cfg:
        raise ValueError("search_params must be a non-empty list in engine config.")
    for item in search_cfg:
        item["params"] = dict(search_params)


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def run_sampling(run_tag: str, cfg: dict[str, Any]) -> Path:
    script = SCRIPT_DIR / "sampling" / "generate_sampled_dataset.py"
    cmd = [
        "python3",
        str(script),
        "--source-path",
        str(cfg["source_path"]),
        "--sample-ratio",
        str(cfg["sample_ratio"]),
        "--sampling-strategy",
        str(cfg["sampling_strategy"]),
        "--kmeans-clusters",
        str(cfg["kmeans_clusters"]),
        "--kmeans-batch-size",
        str(cfg["kmeans_batch_size"]),
        "--seed",
        str(cfg["seed"]),
        "--output-name",
        run_tag,
        "--dataset-name-prefix",
        f"new-adapt-{run_tag}",
        "--neighbors-top-k",
        str(cfg["neighbors_top_k"]),
        "--vector-size",
        str(cfg["vector_size"]),
    ]
    distance = str(cfg["distance"]).strip()
    if distance:
        cmd.extend(["--distance", distance])
    run_command(cmd, cwd=SCRIPT_DIR / "sampling")

    sample_info = BENCHMARK_ROOT / "datasets" / "new_adapt" / run_tag / "sample_info.json"
    if not sample_info.exists():
        raise FileNotFoundError(f"sample_info.json not found: {sample_info}")
    return sample_info


def run_benchmark_once(
    engine_name: str,
    dataset_name: str,
    dataset_path: str,
    vector_size: int,
    distance: str,
    host: str,
    result_meta_path: Path,
) -> Path:
    script = SCRIPT_DIR / "run_custom_benchmark.py"
    cmd = [
        "python3",
        str(script),
        "--benchmark-root",
        str(BENCHMARK_ROOT),
        "--engine-name",
        engine_name,
        "--dataset-name",
        dataset_name,
        "--dataset-path",
        dataset_path,
        "--vector-size",
        str(vector_size),
        "--distance",
        distance,
        "--host",
        host,
        "--result-json",
        str(result_meta_path),
    ]
    run_command(cmd, cwd=SCRIPT_DIR)
    meta = load_json(result_meta_path)
    result_file = Path(meta["result_file"]).resolve()
    if not result_file.exists():
        raise FileNotFoundError(f"Search result file not found: {result_file}")
    return result_file


def extract_metrics(result_json_path: Path) -> tuple[float | None, float | None, float | None]:
    payload = load_json(result_json_path)
    results = payload.get("results", {})
    rps = results.get("rps")
    p95 = results.get("p95_time")
    recall = results.get("mean_precisions")
    return (
        float(rps) if rps is not None else None,
        float(p95) if p95 is not None else None,
        float(recall) if recall is not None else None,
    )


def to_csv(records: list[SweepRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "combo_id",
                "run_tag",
                "index_type",
                "index_params",
                "search_params",
                "rps",
                "p95_time",
                "recall",
                "result_file",
                "status",
                "error",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.combo_id,
                    rec.run_tag,
                    rec.index_type,
                    json.dumps(rec.index_params, ensure_ascii=False),
                    json.dumps(rec.search_params, ensure_ascii=False),
                    rec.rps,
                    rec.p95_time,
                    rec.recall,
                    rec.result_file,
                    rec.status,
                    rec.error,
                ]
            )


def validate_config(cfg: dict[str, Any]) -> None:
    if not (0.0 < float(cfg["sample_ratio"]) <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")
    if int(cfg["top_n"]) <= 0:
        raise ValueError("top_n must be positive.")


def main() -> None:
    cfg = deepcopy(CONFIG)
    validate_config(cfg)

    run_tag = str(cfg["run_tag"]).strip() or f"sampling-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = BENCHMARK_ROOT / "datasets" / "new_adapt" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_info_path = run_sampling(run_tag=run_tag, cfg=cfg)
    sample_info = load_json(sample_info_path)
    sampled_dataset_name = sample_info["sampled_dataset_name"]
    sampled_dir = sample_info["sampled_dir"]
    sampled_vector_size = int(sample_info["vector_size"])
    sampled_distance = str(sample_info["distance"])

    index_grid = normalize_grid(cfg["index_param_grid"], "index_param_grid")
    search_grid = normalize_grid(cfg["search_param_grid"], "search_param_grid")
    index_combos = cartesian_product(index_grid)
    search_combos = cartesian_product(search_grid)
    total_combos = len(index_combos) * len(search_combos)

    config_json_value = str(cfg["config_json"]).strip()
    config_json_path = Path(config_json_value) if config_json_value else DEFAULT_CONFIG_JSON
    if not config_json_path.is_absolute():
        config_json_path = (SCRIPT_DIR / config_json_path).resolve()

    configs = load_config(config_json_path)
    original_configs = deepcopy(configs)
    engine_cfg = find_engine(configs, str(cfg["engine_name"]))
    records: list[SweepRecord] = []

    try:
        seq = 0
        for index_params in index_combos:
            for search_params in search_combos:
                seq += 1
                combo_result_meta = run_dir / "combo_results" / f"combo-{seq:04d}.json"
                print(
                    f"[{seq}/{total_combos}] index_type={cfg['index_type']}, "
                    f"index_params={index_params}, search_params={search_params}"
                )
                set_engine_params(
                    engine_cfg=engine_cfg,
                    index_type=str(cfg["index_type"]),
                    index_params=index_params,
                    search_params=search_params,
                )
                save_config(config_json_path, configs)
                try:
                    result_file = run_benchmark_once(
                        engine_name=str(cfg["engine_name"]),
                        dataset_name=sampled_dataset_name,
                        dataset_path=sampled_dir,
                        vector_size=sampled_vector_size,
                        distance=sampled_distance,
                        host=str(cfg["host"]),
                        result_meta_path=combo_result_meta,
                    )
                    rps, p95, recall = extract_metrics(result_file)
                    records.append(
                        SweepRecord(
                            combo_id=seq,
                            run_tag=run_tag,
                            index_type=str(cfg["index_type"]),
                            index_params=index_params,
                            search_params=search_params,
                            rps=rps,
                            p95_time=p95,
                            recall=recall,
                            result_file=str(result_file),
                            status="ok",
                            error="",
                        )
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    records.append(
                        SweepRecord(
                            combo_id=seq,
                            run_tag=run_tag,
                            index_type=str(cfg["index_type"]),
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
                    if not bool(cfg["continue_on_error"]):
                        raise
    finally:
        save_config(config_json_path, original_configs)

    full_results_path = run_dir / "sampling_index_sweep_results.json"
    save_json(full_results_path, [asdict(r) for r in records])
    to_csv(records, run_dir / "sampling_index_sweep_results.csv")

    valid_rows = [r for r in records if r.status == "ok" and r.recall is not None and r.p95_time is not None]
    candidates = [r for r in valid_rows if float(r.recall) >= float(cfg["recall_threshold"])]
    candidates.sort(key=lambda x: (float(x.p95_time), -(float(x.rps) if x.rps is not None else 0.0)))
    topn = candidates[: int(cfg["top_n"])]

    summary = {
        "run_tag": run_tag,
        "sample_info": str(sample_info_path),
        "full_results_json": str(full_results_path),
        "total_combos": total_combos,
        "ok_combos": sum(1 for r in records if r.status == "ok"),
        "failed_combos": sum(1 for r in records if r.status != "ok"),
        "recall_threshold": float(cfg["recall_threshold"]),
        "top_n": int(cfg["top_n"]),
        "top_configs": [asdict(r) for r in topn],
    }
    summary_path = run_dir / "topn_by_recall_and_p95.json"
    save_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved full results: {full_results_path}")
    print(f"Saved top-N summary: {summary_path}")


if __name__ == "__main__":
    main()
