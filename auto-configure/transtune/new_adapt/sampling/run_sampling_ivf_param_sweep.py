#!/usr/bin/env python3
"""Full-factorial sweep for sampling benchmark parameters."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
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
            "Sweep SAMPLE_RATIO x nlist x nprobe (full factorial) and export "
            "Original vs Sampled metrics to xlsx."
        )
    )
    parser.add_argument(
        "--server-path",
        default="milvus-single-node",
        help="Server path argument for run_sampling_benchmark.sh",
    )
    parser.add_argument(
        "--engine-name",
        default="milvus-p10",
        help="Engine name argument for run_sampling_benchmark.sh",
    )
    parser.add_argument(
        "--source-dataset",
        default="glove-100-angular",
        help="Source dataset argument for run_sampling_benchmark.sh",
    )
    parser.add_argument(
        "--sampling-script",
        default="run_sampling_benchmark.sh",
        help="Path to run_sampling_benchmark.sh (absolute or relative to this file)",
    )
    parser.add_argument(
        "--config-json",
        default="../../../../vector-db-benchmark-master/experiments/configurations/milvus-single-node.json",
        help="Path to engine configuration json file.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="sampling_param_sweep_results.xlsx",
        help="Output Excel (.xlsx) path.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue subsequent runs if one run fails.",
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


def find_engine(configs: list[dict[str, Any]], engine_name: str) -> dict[str, Any]:
    for cfg in configs:
        if cfg.get("name") == engine_name:
            return cfg
    raise KeyError(f"Engine '{engine_name}' not found in configuration json.")


def get_nlist(engine_cfg: dict[str, Any]) -> int:
    return int(engine_cfg["upload_params"]["index_params"]["nlist"])


def set_nlist(engine_cfg: dict[str, Any], nlist: int) -> None:
    engine_cfg["upload_params"]["index_params"]["nlist"] = int(nlist)


def get_nprobe(engine_cfg: dict[str, Any]) -> int:
    search_params = engine_cfg.get("search_params", [])
    if not search_params:
        raise ValueError("search_params is empty; cannot read nprobe.")
    return int(search_params[0]["params"]["nprobe"])


def set_nprobe(engine_cfg: dict[str, Any], nprobe: int) -> None:
    search_params = engine_cfg.get("search_params", [])
    if not search_params:
        raise ValueError("search_params is empty; cannot set nprobe.")
    for item in search_params:
        item.setdefault("params", {})
        item["params"]["nprobe"] = int(nprobe)


def save_config(config_path: Path, configs: list[dict[str, Any]]) -> None:
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(configs, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def extract_metrics(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    original = payload["original"]["metrics"]
    sampled = payload["sampled"]["metrics"]
    return original, sampled


def get_project_root(sampling_script: Path) -> Path:
    return sampling_script.parent.parent.parent.parent.parent


def get_run_dir(project_root: Path, run_tag: str) -> Path:
    return (
        project_root
        / "vector-db-benchmark-master"
        / "datasets"
        / "new_adapt"
        / run_tag
    )


def get_new_adapt_root(project_root: Path) -> Path:
    return project_root / "vector-db-benchmark-master" / "datasets" / "new_adapt"


def run_once(
    sampling_script: Path,
    server_path: str,
    engine_name: str,
    source_dataset: str,
    run_tag: str,
    sample_ratio: float,
) -> Path:
    env = os.environ.copy()
    env["RUN_TAG"] = run_tag
    env["SAMPLE_RATIO"] = str(sample_ratio)
    cmd = [
        "bash",
        str(sampling_script),
        server_path,
        engine_name,
        source_dataset,
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(sampling_script.parent))
    project_root = get_project_root(sampling_script)
    summary = get_run_dir(project_root, run_tag) / "perf_compare_sampling_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Summary not found: {summary}")
    return summary


def make_run_tag(sample_ratio: float, nlist: int, nprobe: int, seq: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_ratio = f"{sample_ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"sample-sweep-r{safe_ratio}-nl{nlist}-np{nprobe}-{seq:04d}-{ts}"


def cleanup_run_dir(run_dir: Path, new_adapt_root: Path) -> None:
    run_dir_resolved = run_dir.resolve()
    new_adapt_root_resolved = new_adapt_root.resolve()
    if new_adapt_root_resolved not in run_dir_resolved.parents:
        raise ValueError(f"Refuse to clean non-new_adapt path: {run_dir_resolved}")
    if run_dir_resolved.exists():
        shutil.rmtree(run_dir_resolved, ignore_errors=True)


def write_xlsx(rows: list[MetricRow], out_path: Path) -> None:
    try:
        from openpyxl import Workbook  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "openpyxl is required to write xlsx. Install with: pip install openpyxl"
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "sampling_sweep"
    headers = [
        "experiment_group",
        "parameter_name",
        "parameter_value",
        "sample_ratio",
        "nlist",
        "nprobe",
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

    sample_ratio_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    nlist_values = list(range(50, 501, 50))
    nprobe_values = list(range(10, 51, 5))
    total_runs = len(sample_ratio_values) * len(nlist_values) * len(nprobe_values)

    configs = load_config(config_path)
    original_configs = deepcopy(configs)
    engine_cfg = find_engine(configs, args.engine_name)
    project_root = get_project_root(sampling_script)
    new_adapt_root = get_new_adapt_root(project_root)

    rows: list[MetricRow] = []

    # Create/initialize excel early so long runs always have a checkpoint file.
    write_xlsx(rows, output_xlsx)

    def append_fail(
        group: str,
        param_name: str,
        param_value: float | int,
        sample_ratio: float,
        nlist: int,
        nprobe: int,
        run_tag: str,
        err: Exception,
    ) -> None:
        rows.append(
            MetricRow(
                experiment_group=group,
                parameter_name=param_name,
                parameter_value=float(param_value),
                sample_ratio=sample_ratio,
                nlist=nlist,
                nprobe=nprobe,
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
        for ratio in sample_ratio_values:
            for nlist in nlist_values:
                for nprobe in nprobe_values:
                    run_index += 1
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"SAMPLE_RATIO={ratio}, nlist={nlist}, nprobe={nprobe}"
                    )
                    run_tag = make_run_tag(ratio, nlist, nprobe, run_index)
                    run_dir = get_run_dir(project_root, run_tag)
                    # Cleanup stale leftovers for the same run tag before starting.
                    cleanup_run_dir(run_dir, new_adapt_root)

                    set_nlist(engine_cfg, nlist)
                    set_nprobe(engine_cfg, nprobe)
                    save_config(config_path, configs)

                    try:
                        summary_path = run_once(
                            sampling_script=sampling_script,
                            server_path=args.server_path,
                            engine_name=args.engine_name,
                            source_dataset=args.source_dataset,
                            run_tag=run_tag,
                            sample_ratio=ratio,
                        )
                        original, sampled = extract_metrics(summary_path)
                        rows.append(
                            MetricRow(
                                experiment_group="full_factorial",
                                parameter_name="combination",
                                parameter_value=float(run_index),
                                sample_ratio=ratio,
                                nlist=nlist,
                                nprobe=nprobe,
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
                            group="full_factorial",
                            param_name="combination",
                            param_value=run_index,
                            sample_ratio=ratio,
                            nlist=nlist,
                            nprobe=nprobe,
                            run_tag=run_tag,
                            err=exc,
                        )
                        write_xlsx(rows, output_xlsx)
                        if not args.continue_on_error:
                            raise
                    finally:
                        # Clean generated datasets for this round to avoid disk growth.
                        cleanup_run_dir(run_dir, new_adapt_root)

        print(f"Completed full-factorial runs: {len(rows)}/{total_runs}")

    finally:
        # Always restore original config.
        save_config(config_path, original_configs)

    write_xlsx(rows, output_xlsx)
    print(f"Done. Rows: {len(rows)}")
    print(f"Saved Excel: {output_xlsx}")


if __name__ == "__main__":
    main()
