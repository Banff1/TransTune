# -*- coding: utf-8 -*-
"""Run vector-db-benchmark via run_engine_test.sh (same contract as ottertune RealEnv)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Repository root containing vector-db-benchmark-master and auto-configure
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ENGINE_PATH = REPO_ROOT / "vector-db-benchmark-master" / "run_engine_test.sh"
BENCHMARK_CWD = RUN_ENGINE_PATH.parent


def parse_benchmark_stdout(result_output: str) -> Tuple[float, float, float]:
    """
    Parse summary line from run_engine_test.sh output.

    Returns
    -------
    mean_precision, rps, p95_time
    Order matches ottertune-configure/vdtuner_interface/utils.py (RealEnv).
    """
    lines = result_output.strip().split("\n")
    numeric_values = []
    for line in reversed(lines):
        if "测试结果摘要" in line or "\U0001f4ca" in line or "结果" in line:
            break
        for word in line.strip().split():
            try:
                numeric_values.append(float(word))
            except ValueError:
                continue

    if len(numeric_values) < 3:
        for item in result_output.strip().split():
            try:
                numeric_values.append(float(item))
            except ValueError:
                continue

    if len(numeric_values) < 3:
        raise ValueError(
            f"Could not parse benchmark output (need 3 numbers). Last 500 chars:\n{result_output[-500:]!r}"
        )

    mean_precision = numeric_values[-1]
    p95_time = numeric_values[-3]
    rps = numeric_values[-2]
    return mean_precision, rps, p95_time


def run_milvus_benchmark(
    engine: str,
    process_tag: str,
    dataset: str,
    *,
    use_sudo: bool = True,
    timeout_sec: int = 2000,
) -> Tuple[float, float, float]:
    """
    Deploy current Milvus config (already written by configure.configure_*) and run one benchmark.

    Returns mean_precision, rps, p95_latency.
    """
    if not RUN_ENGINE_PATH.is_file():
        raise FileNotFoundError(f"Missing benchmark script: {RUN_ENGINE_PATH}")

    cmd = ["timeout", str(timeout_sec), str(RUN_ENGINE_PATH), engine, process_tag, dataset]
    if use_sudo:
        cmd = ["sudo"] + cmd

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        cwd=str(BENCHMARK_CWD),
    )
    output_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark failed with exit code {proc.returncode}")

    return parse_benchmark_stdout("".join(output_lines))
