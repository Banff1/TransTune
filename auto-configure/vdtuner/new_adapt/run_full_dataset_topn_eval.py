#!/usr/bin/env python3
"""Evaluate sampled-sweep Top-N configs on full dataset and pick the best one.

Usage:
1) Edit CONFIG below.
2) Run: python3 auto-configure/vdtuner/new_adapt/run_full_dataset_topn_eval.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import subprocess


# ===================== User Config =====================
CONFIG: dict[str, Any] = {
    # Input results from run_sampling_index_sweep.py
    "sampling_results_json": (
        "/path/to/TransTune/"
        "vector-db-benchmark-master/datasets/new_adapt/"
        "sampling-sweep-20260530-230020/sampling_index_sweep_results.json"
    ),
    "sample_info_json": (
        "/path/to/TransTune/"
        "vector-db-benchmark-master/datasets/new_adapt/"
        "sampling-sweep-20260530-230020/sample_info.json"
    ),

    # Candidate selection on sampled results:
    # keep records with recall >= threshold, sort by p95 ascending, take top_n.
    "recall_threshold": 0.85,
    "top_n": 5,

    # Benchmark engine settings
    "engine_name": "milvus-p10",
    "host": "127.0.0.1",
    "config_json": "",  # empty means default milvus-single-node config

    # Full dataset override (empty = infer from sample_info.json)
    "full_dataset_name": "",
    "full_dataset_path": "",
    "full_vector_size": 0,
    "full_distance": "",

    # Final best-config rule on full dataset
    # supported: "rps" (higher better), "p95_time" (lower better)
    "best_metric": "rps",
    # Optional recall guard on full-dataset retest, <=0 means disabled
    "full_recall_threshold": 0.0,

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
class EvalRecord:
    rank_in_sampled_topn: int
    source_combo_id: int
    index_type: str
    index_params: dict[str, Any]
    search_params: dict[str, Any]
    sampled_recall: float | None
    sampled_p95_time: float | None
    sampled_rps: float | None
    full_rps: float | None
    full_p95_time: float | None
    full_recall: float | None
    full_result_file: str
    status: str
    error: str


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


def choose_topn_from_sampled(
    sampled_rows: list[dict[str, Any]],
    recall_threshold: float,
    top_n: int,
) -> list[dict[str, Any]]:
    ok_rows = [r for r in sampled_rows if r.get("status") == "ok"]
    filtered = [
        r
        for r in ok_rows
        if r.get("recall") is not None
        and r.get("p95_time") is not None
        and float(r["recall"]) >= recall_threshold
    ]
    filtered.sort(key=lambda x: (float(x["p95_time"]), -(float(x.get("rps") or 0.0))))
    return filtered[:top_n]


def resolve_full_dataset(cfg: dict[str, Any], sample_info: dict[str, Any]) -> dict[str, Any]:
    name = str(cfg["full_dataset_name"]).strip()
    path = str(cfg["full_dataset_path"]).strip()
    vector_size = int(cfg["full_vector_size"])
    distance = str(cfg["full_distance"]).strip()

    if not name:
        name = str(sample_info.get("source_dataset_name") or sample_info.get("source_label") or "full-dataset")
    if not distance:
        distance = str(sample_info.get("distance") or "cosine")
    if vector_size <= 0:
        vector_size = int(sample_info.get("vector_size") or 0)

    # For h5 source resolved from datasets.json name, do not force dataset_path.
    source_kind = str(sample_info.get("source_kind") or "")
    if not path and source_kind == "jsonl":
        path = str(sample_info.get("source_dir") or "")

    if vector_size <= 0:
        raise ValueError("full vector_size cannot be resolved; set CONFIG['full_vector_size'].")

    return {
        "dataset_name": name,
        "dataset_path": path,
        "vector_size": vector_size,
        "distance": distance,
    }


def pick_best(records: list[EvalRecord], best_metric: str, full_recall_threshold: float) -> EvalRecord | None:
    ok = [r for r in records if r.status == "ok" and r.full_rps is not None and r.full_p95_time is not None]
    if full_recall_threshold > 0:
        ok = [r for r in ok if r.full_recall is not None and float(r.full_recall) >= full_recall_threshold]
    if not ok:
        return None

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
    return ok[0]


def validate_config(cfg: dict[str, Any]) -> None:
    if int(cfg["top_n"]) <= 0:
        raise ValueError("top_n must be positive.")
    if float(cfg["recall_threshold"]) < 0 or float(cfg["recall_threshold"]) > 1:
        raise ValueError("recall_threshold must be in [0,1].")
    if str(cfg["best_metric"]) not in {"rps", "p95_time"}:
        raise ValueError("best_metric must be 'rps' or 'p95_time'.")


def main() -> None:
    cfg = deepcopy(CONFIG)
    validate_config(cfg)

    run_tag = str(cfg["run_tag"]).strip() or f"full-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = BENCHMARK_ROOT / "datasets" / "new_adapt" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_rows = load_json(Path(str(cfg["sampling_results_json"])).resolve())
    if not isinstance(sampled_rows, list):
        raise ValueError("sampling_results_json must be a JSON list.")
    sample_info = load_json(Path(str(cfg["sample_info_json"])).resolve())
    full_dataset = resolve_full_dataset(cfg, sample_info)

    candidates = choose_topn_from_sampled(
        sampled_rows=sampled_rows,
        recall_threshold=float(cfg["recall_threshold"]),
        top_n=int(cfg["top_n"]),
    )
    if not candidates:
        raise RuntimeError(
            "No sampled candidates satisfy recall threshold. "
            "Lower CONFIG['recall_threshold'] or re-run sampled sweep with higher recall settings."
        )

    config_json_value = str(cfg["config_json"]).strip()
    config_json_path = Path(config_json_value) if config_json_value else DEFAULT_CONFIG_JSON
    if not config_json_path.is_absolute():
        config_json_path = (SCRIPT_DIR / config_json_path).resolve()

    configs = load_config(config_json_path)
    original_configs = deepcopy(configs)
    engine_cfg = find_engine(configs, str(cfg["engine_name"]))
    records: list[EvalRecord] = []

    try:
        for rank, cand in enumerate(candidates, start=1):
            result_meta_path = out_dir / "full_eval_results" / f"rank-{rank:02d}.json"
            index_type = str(cand["index_type"])
            index_params = dict(cand.get("index_params") or {})
            search_params = dict(cand.get("search_params") or {})
            print(
                f"[{rank}/{len(candidates)}] full-eval combo_id={cand.get('combo_id')}, "
                f"index_type={index_type}, index_params={index_params}, search_params={search_params}"
            )

            set_engine_params(
                engine_cfg=engine_cfg,
                index_type=index_type,
                index_params=index_params,
                search_params=search_params,
            )
            save_config(config_json_path, configs)
            try:
                result_file = run_benchmark_once(
                    engine_name=str(cfg["engine_name"]),
                    dataset_name=str(full_dataset["dataset_name"]),
                    dataset_path=str(full_dataset["dataset_path"]),
                    vector_size=int(full_dataset["vector_size"]),
                    distance=str(full_dataset["distance"]),
                    host=str(cfg["host"]),
                    result_meta_path=result_meta_path,
                )
                rps, p95, recall = extract_metrics(result_file)
                records.append(
                    EvalRecord(
                        rank_in_sampled_topn=rank,
                        source_combo_id=int(cand.get("combo_id", -1)),
                        index_type=index_type,
                        index_params=index_params,
                        search_params=search_params,
                        sampled_recall=float(cand["recall"]) if cand.get("recall") is not None else None,
                        sampled_p95_time=float(cand["p95_time"]) if cand.get("p95_time") is not None else None,
                        sampled_rps=float(cand["rps"]) if cand.get("rps") is not None else None,
                        full_rps=rps,
                        full_p95_time=p95,
                        full_recall=recall,
                        full_result_file=str(result_file),
                        status="ok",
                        error="",
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                records.append(
                    EvalRecord(
                        rank_in_sampled_topn=rank,
                        source_combo_id=int(cand.get("combo_id", -1)),
                        index_type=index_type,
                        index_params=index_params,
                        search_params=search_params,
                        sampled_recall=float(cand["recall"]) if cand.get("recall") is not None else None,
                        sampled_p95_time=float(cand["p95_time"]) if cand.get("p95_time") is not None else None,
                        sampled_rps=float(cand["rps"]) if cand.get("rps") is not None else None,
                        full_rps=None,
                        full_p95_time=None,
                        full_recall=None,
                        full_result_file="",
                        status="failed",
                        error=str(exc),
                    )
                )
    finally:
        save_config(config_json_path, original_configs)

    best = pick_best(
        records=records,
        best_metric=str(cfg["best_metric"]),
        full_recall_threshold=float(cfg["full_recall_threshold"]),
    )

    eval_json = out_dir / "full_dataset_topn_eval_results.json"
    save_json(eval_json, [asdict(r) for r in records])

    summary = {
        "run_tag": run_tag,
        "sampling_results_json": str(Path(str(cfg["sampling_results_json"])).resolve()),
        "sample_info_json": str(Path(str(cfg["sample_info_json"])).resolve()),
        "candidate_recall_threshold": float(cfg["recall_threshold"]),
        "candidate_top_n": int(cfg["top_n"]),
        "full_dataset": full_dataset,
        "best_metric": str(cfg["best_metric"]),
        "full_recall_threshold": float(cfg["full_recall_threshold"]),
        "evaluated_candidates": len(records),
        "ok_candidates": sum(1 for r in records if r.status == "ok"),
        "failed_candidates": sum(1 for r in records if r.status != "ok"),
        "best_config": asdict(best) if best is not None else None,
        "all_eval_results_json": str(eval_json),
    }
    summary_json = out_dir / "best_config_on_full_dataset.json"
    save_json(summary_json, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved eval results: {eval_json}")
    print(f"Saved best config summary: {summary_json}")


if __name__ == "__main__":
    main()
