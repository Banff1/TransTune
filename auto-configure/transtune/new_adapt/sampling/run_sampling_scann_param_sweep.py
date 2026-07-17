#!/usr/bin/env python3
"""Full-factorial sweep for SCANN parameters with sampling benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MetricRow:
    experiment_group: str
    parameter_name: str
    parameter_value: float
    sample_ratio: float
    nlist: int
    nprobe: int
    reorder_k: int
    original_rps: float | None
    original_p95_time: float | None
    original_mean_precisions: float | None
    sampled_rps: float | None
    sampled_p95_time: float | None
    sampled_mean_precisions: float | None
    run_tag: str
    summary_json: str
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SAMPLE_RATIO x nlist x nprobe x reorder_k for SCANN and "
            "export Original vs Sampled metrics to xlsx."
        )
    )
    parser.add_argument("--server-path", default="milvus-single-node")
    parser.add_argument("--engine-name", default="milvus-p10")
    parser.add_argument("--source-dataset", default="glove-100-angular")
    parser.add_argument("--sampling-script", default="run_sampling_benchmark.sh")
    parser.add_argument(
        "--config-json",
        default="../../../../vector-db-benchmark-master/experiments/configurations/milvus-single-node.json",
    )
    parser.add_argument(
        "--output-xlsx",
        default="sampling_new_scann_param_sweep_results.xlsx",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue subsequent runs if one run fails.",
    )
    parser.add_argument(
        "--reuse-sampled-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one generated sampled dataset per sample_ratio.",
    )
    return parser.parse_args()


def resolve_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_config(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Config json must be a list: {config_path}")
    return data


def save_config(config_path: Path, configs: list[dict[str, Any]]) -> None:
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(configs, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def find_engine(configs: list[dict[str, Any]], engine_name: str) -> dict[str, Any]:
    for cfg in configs:
        if cfg.get("name") == engine_name:
            return cfg
    raise KeyError(f"Engine '{engine_name}' not found in configuration json.")


def set_scann_params(
    engine_cfg: dict[str, Any],
    nlist: int,
    nprobe: int,
    reorder_k: int,
) -> None:
    upload_params = engine_cfg.setdefault("upload_params", {})
    upload_params["index_type"] = "SCANN"
    upload_params["index_params"] = {
        "nlist": int(nlist),
    }

    search_params = engine_cfg.get("search_params", [])
    if not search_params:
        raise ValueError("search_params is empty; cannot set nprobe/reorder_k.")
    for item in search_params:
        item.setdefault("params", {})
        item["params"]["nprobe"] = int(nprobe)
        item["params"]["reorder_k"] = int(reorder_k)


def extract_metrics(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["original"]["metrics"], payload["sampled"]["metrics"]


def get_project_root(sampling_script: Path) -> Path:
    return sampling_script.parent.parent.parent.parent.parent


def get_new_adapt_root(project_root: Path) -> Path:
    return project_root / "vector-db-benchmark-master" / "datasets" / "new_adapt"


def get_run_dir(project_root: Path, run_tag: str) -> Path:
    return get_new_adapt_root(project_root) / run_tag


def cleanup_run_dir(run_dir: Path, new_adapt_root: Path) -> None:
    run_dir_resolved = run_dir.resolve()
    root_resolved = new_adapt_root.resolve()
    if root_resolved not in run_dir_resolved.parents:
        raise ValueError(f"Refuse to clean non-new_adapt path: {run_dir_resolved}")
    if run_dir_resolved.exists():
        shutil.rmtree(run_dir_resolved, ignore_errors=True)


def run_once(
    sampling_script: Path,
    server_path: str,
    engine_name: str,
    source_dataset: str,
    run_tag: str,
    sample_ratio: float,
    sample_reuse_tag: str | None = None,
    reuse_sampled_dataset: bool = True,
) -> Path:
    env = os.environ.copy()
    env["RUN_TAG"] = run_tag
    env["SAMPLE_RATIO"] = str(sample_ratio)
    if sample_reuse_tag:
        env["SAMPLE_REUSE_TAG"] = sample_reuse_tag
    env["REUSE_SAMPLED_DATASET"] = "1" if reuse_sampled_dataset else "0"
    cmd = [
        "bash",
        str(sampling_script),
        server_path,
        engine_name,
        source_dataset,
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(sampling_script.parent))
    project_root = get_project_root(sampling_script)
    summary_path = get_run_dir(project_root, run_tag) / "perf_compare_sampling_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    return summary_path


def make_run_tag(sample_ratio: float, nlist: int, nprobe: int, reorder_k: int, seq: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_ratio = f"{sample_ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"sample-scann-r{safe_ratio}-nl{nlist}-np{nprobe}-rk{reorder_k}-{seq:04d}-{ts}"


def make_sample_cache_tag(sample_ratio: float) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_ratio = f"{sample_ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"sampled-cache-r{safe_ratio}-{ts}"


def write_xlsx(rows: list[MetricRow], out_path: Path) -> None:
    try:
        from openpyxl import Workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to write xlsx. Install with: pip install openpyxl"
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "sampling_scann_sweep"
    headers = [
        "experiment_group",
        "parameter_name",
        "parameter_value",
        "sample_ratio",
        "nlist",
        "nprobe",
        "reorder_k",
        "original_rps",
        "original_p95_time",
        "original_mean_precisions",
        "sampled_rps",
        "sampled_p95_time",
        "sampled_mean_precisions",
        "run_tag",
        "summary_json",
        "status",
        "error",
    ]
    ws.append(headers)
    for r in rows:
        ws.append(
            [
                r.experiment_group,
                r.parameter_name,
                r.parameter_value,
                r.sample_ratio,
                r.nlist,
                r.nprobe,
                r.reorder_k,
                r.original_rps,
                r.original_p95_time,
                r.original_mean_precisions,
                r.sampled_rps,
                r.sampled_p95_time,
                r.sampled_mean_precisions,
                r.run_tag,
                r.summary_json,
                r.status,
                r.error,
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    sampling_script = resolve_path(args.sampling_script, script_dir)
    config_path = resolve_path(args.config_json, script_dir)
    output_xlsx = resolve_path(args.output_xlsx, script_dir)

    # sample_ratio_values = [0.01, 0.03, 0.05, 0.08, 0.10]
    sample_ratio_values = [0.01, 0.03, 0.05]
    nlist_values = list(range(100, 401, 75))
    nprobe_values = list(range(10, 111, 50))
    reorder_k_values = list(range(100, 201, 50))
    total_runs = (
        len(sample_ratio_values)
        * len(nlist_values)
        * len(nprobe_values)
        * len(reorder_k_values)
    )

    configs = load_config(config_path)
    original_configs = deepcopy(configs)
    engine_cfg = find_engine(configs, args.engine_name)

    project_root = get_project_root(sampling_script)
    new_adapt_root = get_new_adapt_root(project_root)
    rows: list[MetricRow] = []

    # Initialize xlsx early for checkpointing.
    write_xlsx(rows, output_xlsx)

    def append_fail(
        sample_ratio: float,
        nlist: int,
        nprobe: int,
        reorder_k: int,
        run_tag: str,
        err: Exception,
        seq: int,
    ) -> None:
        rows.append(
            MetricRow(
                experiment_group="scann_full_factorial",
                parameter_name="combination",
                parameter_value=float(seq),
                sample_ratio=sample_ratio,
                nlist=nlist,
                nprobe=nprobe,
                reorder_k=reorder_k,
                original_rps=None,
                original_p95_time=None,
                original_mean_precisions=None,
                sampled_rps=None,
                sampled_p95_time=None,
                sampled_mean_precisions=None,
                run_tag=run_tag,
                summary_json="",
                status="failed",
                error=str(err),
            )
        )

    def _handle_termination(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)

    try:
        run_index = 0
        sample_cache_tags: dict[float, str] = {}
        for sample_ratio in sample_ratio_values:
            if sample_ratio not in sample_cache_tags:
                sample_cache_tags[sample_ratio] = make_sample_cache_tag(sample_ratio)
            ratio_cache_tag = sample_cache_tags[sample_ratio]
            for nlist in nlist_values:
                for nprobe in nprobe_values:
                    for reorder_k in reorder_k_values:
                        run_index += 1
                        print(
                            f"[{run_index}/{total_runs}] "
                            f"SAMPLE_RATIO={sample_ratio}, nlist={nlist}, "
                            f"nprobe={nprobe}, reorder_k={reorder_k}"
                        )
                        run_tag = make_run_tag(
                            sample_ratio, nlist, nprobe, reorder_k, run_index
                        )
                        sample_reuse_tag = ratio_cache_tag if args.reuse_sampled_dataset else run_tag
                        run_dir = get_run_dir(project_root, run_tag)
                        cleanup_run_dir(run_dir, new_adapt_root)

                        set_scann_params(
                            engine_cfg=engine_cfg,
                            nlist=nlist,
                            nprobe=nprobe,
                            reorder_k=reorder_k,
                        )
                        save_config(config_path, configs)

                        try:
                            summary_path = run_once(
                                sampling_script=sampling_script,
                                server_path=args.server_path,
                                engine_name=args.engine_name,
                                source_dataset=args.source_dataset,
                                run_tag=run_tag,
                                sample_ratio=sample_ratio,
                                sample_reuse_tag=sample_reuse_tag,
                                reuse_sampled_dataset=args.reuse_sampled_dataset,
                            )
                            original, sampled = extract_metrics(summary_path)
                            rows.append(
                                MetricRow(
                                    experiment_group="scann_full_factorial",
                                    parameter_name="combination",
                                    parameter_value=float(run_index),
                                    sample_ratio=sample_ratio,
                                    nlist=nlist,
                                    nprobe=nprobe,
                                    reorder_k=reorder_k,
                                    original_rps=original.get("rps"),
                                    original_p95_time=original.get("p95_time"),
                                    original_mean_precisions=original.get("mean_precisions"),
                                    sampled_rps=sampled.get("rps"),
                                    sampled_p95_time=sampled.get("p95_time"),
                                    sampled_mean_precisions=sampled.get("mean_precisions"),
                                    run_tag=run_tag,
                                    summary_json=str(summary_path),
                                    status="ok",
                                    error="",
                                )
                            )
                            write_xlsx(rows, output_xlsx)
                        except Exception as exc:
                            append_fail(
                                sample_ratio=sample_ratio,
                                nlist=nlist,
                                nprobe=nprobe,
                                reorder_k=reorder_k,
                                run_tag=run_tag,
                                err=exc,
                                seq=run_index,
                            )
                            write_xlsx(rows, output_xlsx)
                            if not args.continue_on_error:
                                raise
                        finally:
                            cleanup_run_dir(run_dir, new_adapt_root)

        print(f"Completed SCANN full-factorial runs: {len(rows)}/{total_runs}")
    finally:
        # Always restore original config.
        save_config(config_path, original_configs)

    write_xlsx(rows, output_xlsx)
    print(f"Done. Rows: {len(rows)}")
    print(f"Saved Excel: {output_xlsx}")


if __name__ == "__main__":
    main()
