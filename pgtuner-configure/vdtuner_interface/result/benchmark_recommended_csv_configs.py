#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deploy recommended CSV configs and run Milvus benchmarks."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
VDIR = REPO / "pgtuner-configure" / "vdtuner_interface"
PCR = REPO / "pgtuner-configure" / "parameter_configuration_recommend"
AUTO = REPO / "auto-configure"

sys.path.insert(0, str(VDIR))
sys.path.insert(0, str(AUTO))

from benchmark_runner import run_milvus_benchmark  # noqa: E402
from configure import configure_index, configure_system, filter_index_rule, filter_system_rule  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402

CSV_PATH = (
    PCR
    / "[128, 256, 256, 64]_[256, 256, 256, 64]_TD3_prior/main/recommend_results"
    / "eval_250_100_128_1e-05_0.0001_0.2_2_1_200_vdb_random-100-match-kw-small-vocab-no-filters_250_150.csv"
)
PRIOR_CONFIG = VDIR / "example_vdtuner_prior.json"
OUT_PATH = Path(__file__).resolve().parent / "benchmark_recommended_tr085_090_095.json"

# user asked 0.86 / 0.90 / 0.95; CSV uses target_recall 0.85 / 0.90 / 0.95
TARGETS = [
    (0.86, 0.85),
    (0.90, 0.90),
    (0.95, 0.95),
]

ENGINE = "milvus-single-node"
PROCESS_TAG = "milvus-p10"
DATASET = "random-100-match-kw-small-vocab-no-filters"


def row_to_assignment(space, row: dict) -> dict:
    raw = np.array([float(row[name]) for name in space.knob_names], dtype=np.float32)
    return space.raw_numpy_row_to_value_dict(raw)


def deploy_config(full_conf: dict) -> None:
    work = dict(full_conf)
    configure_index(*filter_index_rule(work))
    system_conf = {k: full_conf[k] for k in full_conf if "*" in k}
    configure_system(filter_system_rule(system_conf))


def main() -> None:
    space = load_prior_space(None, prior_config_path=PRIOR_CONFIG)
    rows_by_target = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_target[float(row["target_recall"])] = row

    results = []
    for label, target in TARGETS:
        if target not in rows_by_target:
            raise SystemExit(f"Missing target_recall={target} in {CSV_PATH}")
        row = rows_by_target[target]
        assignment = row_to_assignment(space, row)
        full_conf = space.build_full_milvus_config(assignment)

        print(f"\n{'='*60}", flush=True)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] label={label} target_recall={target}", flush=True)
        print(f"assignment={assignment}", flush=True)

        deploy_config(full_conf)

        recall, rps, p95 = run_milvus_benchmark(
            ENGINE, PROCESS_TAG, DATASET, use_sudo=True, timeout_sec=3600
        )
        entry = {
            "label_recall": label,
            "target_recall": target,
            "assignment": assignment,
            "recall": recall,
            "rps": rps,
            "p95_latency_s": p95,
            "p95_latency_ms": p95 * 1000,
        }
        results.append(entry)
        print(
            f"RESULT label={label}: recall={recall:.6f} rps={rps:.2f} p95={p95*1000:.4f}ms",
            flush=True,
        )

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
