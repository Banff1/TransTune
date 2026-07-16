#!/usr/bin/env python3
"""
OtterTune-style Milvus task characterization: same canonical workload + Prometheus observation.

Workflow (per task):
  1. (Optional) Start a subprocess running the same benchmark / load script as other tasks.
  2. Poll Prometheus every --interval for --samples (or until workload exits + optional tail).
  3. Aggregate metrics (mean/std) → TaskProfile JSON.
  4. Compare tasks: pairwise distance matrix (euclidean or cosine).

Example (from repo root ``ottertune-configure/``):

  python -m milvus_ottertune.run_task_characterization characterize \\
    --prometheus http://127.0.0.1:9090 \\
    --task-id glove-100-angular \\
    --workload-cmd "timeout 600 ./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular" \\
    --workload-cwd /path/to/vector-db-benchmark-master \\
    --profiles-dir ./milvus_ottertune/task_profiles \\
    --samples 30 --interval 10

  # Or: cd milvus_ottertune && python run_task_characterization.py ...

  python -m milvus_ottertune.run_task_characterization compare --profiles-dir ./milvus_ottertune/task_profiles
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import requests
import typer

# This file lives in milvus_ottertune/; parent is ottertune-configure (must be on sys.path for imports).
_MILVUS_OT_DIR = Path(__file__).resolve().parent
_OTTER_CONFIGURE_ROOT = _MILVUS_OT_DIR.parent
if str(_OTTER_CONFIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_OTTER_CONFIGURE_ROOT))

from milvus_ottertune.milvus_prometheus_collector import MilvusMetricsCollector
from milvus_ottertune.task_similarity import (
    TaskProfile,
    aggregate_observation,
    load_profiles_from_dir,
    pairwise_distance_matrix,
    save_profile,
)

app = typer.Typer(help="Milvus Prometheus task characterization & similarity")
LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _check_prometheus_reachable(base_url: str, verify_tls: bool) -> None:
    """
    Fail fast with a clear message if nothing is listening (typical: Prometheus not started).
    """
    base = base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/-/ready", timeout=10, verify=verify_tls)
        if r.status_code == 200:
            return
    except requests.exceptions.ConnectionError as e:
        typer.secho(
            f"无法连接 Prometheus: {base_url}\n"
            f"  [Errno 111] Connection refused 表示当前机器上 **没有进程在监听该地址的端口**（常见是 Prometheus 未启动）。\n\n"
            f"  请先做其一：\n"
            f"  1) 在本机启动 Prometheus，并把端口映射到脚本里用的地址（例如 -p 9090:9090）。\n"
            f"  2) 若 Prometheus 跑在别的机器/容器里，把 --prometheus 改成实际可访问的 URL（如 http://宿主机IP:9090）。\n\n"
            f"  自检命令: curl -sS '{base}/-/ready'  或浏览器打开 Prometheus UI。\n"
            f"  另需在 Prometheus 的 scrape_configs 里配置抓取 Milvus 的 /metrics 端点，否则查询会无数据。\n\n"
            f"  底层错误: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from e
    except requests.exceptions.RequestException as e:
        typer.secho(f"访问 Prometheus 失败 ({base_url}): {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    typer.secho(
        f"Prometheus 返回非 200 (/-/ready): {base_url} -> HTTP {r.status_code}",
        fg=typer.colors.YELLOW,
        err=True,
    )


@app.command()
def characterize(
    prometheus: str = typer.Option(..., help="Prometheus base URL, e.g. http://localhost:9090"),
    task_id: str = typer.Option(..., help="Unique task / workload label (e.g. dataset name)"),
    profiles_dir: Path = typer.Option(Path("./task_profiles"), help="Directory for profile JSON"),
    metrics_json: Optional[Path] = typer.Option(
        None,
        help="Optional JSON: {\"metric_name\": \"promql\", ...} to override defaults",
    ),
    samples: int = typer.Option(30, help="Number of Prometheus polls (observation samples)"),
    interval: float = typer.Option(10.0, help="Seconds between polls"),
    workload_cmd: Optional[str] = typer.Option(
        None,
        help="Shell command for canonical load; if set, observation runs in parallel until it exits",
    ),
    workload_cwd: Optional[Path] = typer.Option(
        None,
        help="Working directory for workload subprocess",
    ),
    use_sudo_workload: bool = typer.Option(
        False,
        help="Prepend sudo to workload (only if your script needs it)",
    ),
    tail_after_workload_sec: float = typer.Option(
        0.0,
        help="After workload exits, keep polling for this many seconds (0 = stop immediately)",
    ),
    no_tls_verify: bool = typer.Option(False, help="Disable TLS verify for Prometheus"),
    skip_prometheus_check: bool = typer.Option(
        False,
        help="Do not verify Prometheus is reachable before starting (not recommended)",
    ),
    no_fill_missing: bool = typer.Option(
        False,
        help="Only store metrics that returned data from Prometheus (profile keys vary by run)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Run (optional) canonical workload and collect Prometheus snapshots → save TaskProfile.
    """
    _setup_logging(verbose)
    if not skip_prometheus_check:
        _check_prometheus_reachable(prometheus, not no_tls_verify)
    collector = _make_collector(prometheus, metrics_json, no_tls_verify)
    fill_missing = not no_fill_missing

    history: List[dict] = []
    prometheus_hits: Dict[str, int] = defaultdict(int)
    stop_poll = threading.Event()
    workload_rc: List[Optional[int]] = [None]

    def append_snapshot(sample_index: int) -> None:
        snap, fetch_ok = collector.collect_all_metrics_with_fetch_status(
            fill_missing=fill_missing
        )
        for name, ok in fetch_ok.items():
            if ok:
                prometheus_hits[name] += 1
        snap["timestamp_epoch"] = time.time()
        snap["sample_index"] = sample_index
        history.append(snap)

    def poll_loop() -> None:
        idx = 0
        while not stop_poll.is_set() and idx < samples:
            append_snapshot(idx)
            n_m = len(collector.metrics_map)
            n_fields = len(history[-1]) - 2  # exclude timestamp_epoch, sample_index
            LOG.info(
                "Sample %s/%s: %s metric fields (configured=%s)",
                idx + 1,
                samples,
                n_fields,
                n_m,
            )
            idx += 1
            if stop_poll.wait(timeout=interval):
                break

    if workload_cmd:
        cmd = workload_cmd
        if use_sudo_workload:
            cmd = f"sudo {cmd}"
        cwd = str(workload_cwd) if workload_cwd else None
        LOG.info("Starting workload: %s (cwd=%s)", cmd, cwd)
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        poll_thread = threading.Thread(target=poll_loop, daemon=True)
        poll_thread.start()
        _, err = proc.communicate()
        workload_rc[0] = proc.returncode
        if proc.returncode != 0:
            LOG.warning("Workload exited with code %s; stderr (truncated): %s", proc.returncode, (err or b"")[:500])
        # Stop polling as soon as workload exits (avoid sampling idle cluster).
        stop_poll.set()
        poll_thread.join(timeout=30.0)
        if tail_after_workload_sec > 0:
            deadline = time.time() + tail_after_workload_sec
            while time.time() < deadline and len(history) < samples:
                time.sleep(interval)
                append_snapshot(len(history))
    else:
        LOG.info("No workload_cmd: collecting %s samples every %ss", samples, interval)
        poll_loop()

    names, mean_v, std_v, counts = aggregate_observation(history)
    profile = TaskProfile(
        task_id=task_id,
        metric_names=names,
        mean_vector=mean_v,
        std_vector=std_v,
        num_samples=len(history),
        raw_sample_count_per_metric=counts,
        meta={
            "prometheus": prometheus,
            "samples_requested": samples,
            "interval_sec": interval,
            "workload_cmd": workload_cmd,
            "workload_rc": workload_rc[0],
            "fill_missing": fill_missing,
            "raw_sample_count_per_metric_note_zh": (
                "在开启 fill_missing（默认）时，每一轮采集都会为每个指标写入一个数值（含占位 0），"
                "因此 raw_sample_count_per_metric 里各指标通常相同，且等于观测轮数；"
                "这不表示 Prometheus 对每个查询都返回了真实数据。"
            ),
            "prometheus_query_success_samples_per_metric": dict(prometheus_hits),
        },
    )
    out = profiles_dir / f"{_safe_filename(task_id)}.json"
    save_profile(profile, out)
    LOG.info("Saved profile: %s (metrics=%s, rows=%s)", out, len(names), len(history))
    typer.echo(json.dumps({"profile": str(out), "metrics": len(names), "rows": len(history)}, indent=2))


@app.command()
def compare(
    profiles_dir: Path = typer.Option(Path("./task_profiles"), help="Directory with *.json profiles"),
    metric: str = typer.Option("euclidean", help="euclidean or cosine"),
    no_standardize: bool = typer.Option(False, help="Disable cross-task z-score on features"),
    no_std_features: bool = typer.Option(False, help="Do not append std vector to features"),
    pca: int = typer.Option(0, help="If >0 and sklearn installed, PCA n_components before distance"),
    fa: int = typer.Option(
        0,
        help="If >0, use FactorAnalysis (OtterTune-style) instead of PCA; overrides --pca",
    ),
    out_json: Optional[Path] = typer.Option(None, help="Write distance matrix JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Load all profiles in directory and print pairwise distance matrix."""
    _setup_logging(verbose)
    profiles = load_profiles_from_dir(profiles_dir)
    if len(profiles) < 2:
        typer.echo("Need at least 2 profile JSON files in the directory.", err=True)
        raise typer.Exit(1)
    D, ids = pairwise_distance_matrix(
        profiles,
        metric=metric,  # type: ignore[arg-type]
        zscore_across_tasks=not no_standardize,
        include_std_in_features=not no_std_features,
        pca_components=pca if pca > 0 and fa <= 0 else None,
        fa_components=fa if fa > 0 else None,
    )
    typer.echo("Task IDs: " + ", ".join(ids))
    typer.echo("Distance matrix (lower = more similar under same probe workload):")
    typer.echo(_format_matrix(D, ids))
    if out_json:
        payload = {"task_ids": ids, "metric": metric, "matrix": D.tolist()}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out_json}")


def _make_collector(
    prometheus: str,
    metrics_json: Optional[Path],
    no_tls_verify: bool,
) -> MilvusMetricsCollector:
    if metrics_json:
        return MilvusMetricsCollector.from_json(prometheus, metrics_json, verify_tls=not no_tls_verify)
    return MilvusMetricsCollector(prometheus, verify_tls=not no_tls_verify)


def _safe_filename(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:200]


def _format_matrix(D, ids: List[str]) -> str:
    w = max(len(s) for s in ids) if ids else 8
    head = " " * (w + 2) + " ".join(f"{j:>8}" for j in range(len(ids)))
    lines = [head]
    for i, rid in enumerate(ids):
        row = f"{rid:<{w}}  " + " ".join(f"{D[i, j]:8.4f}" for j in range(len(ids)))
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    app()
