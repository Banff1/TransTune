#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deploy selected prior recommendations to Milvus config files and benchmark them.

This script reads the recommendation CSV produced by
`generate_recommended_configurations_vdtuner_prior.py`, selects rows by target_recall
(exact match or nearest), decodes knob values, writes:

- vector-db-benchmark-master/experiments/configurations/milvus-single-node.json
- vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml

then runs one benchmark per selected row and stores measured recall / rps / p95.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent
AUTO_CONFIGURE = REPO_ROOT / "auto-configure"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(AUTO_CONFIGURE) not in sys.path:
    sys.path.insert(0, str(AUTO_CONFIGURE))

from benchmark_runner import run_milvus_benchmark  # noqa: E402
from configure import configure_index, configure_system, filter_index_rule, filter_system_rule  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402


def _parse_target_recalls(text: str) -> List[float]:
    vals = []
    for t in text.split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("--target-recalls cannot be empty")
    return vals


def _select_row(df: pd.DataFrame, target: float, exact: bool) -> pd.Series:
    if "target_recall" not in df.columns:
        raise ValueError("recommend CSV must contain column: target_recall")
    if exact:
        m = np.isclose(df["target_recall"].astype(float).values, target, atol=1e-9)
        if not m.any():
            raise ValueError(f"target_recall={target} not found in CSV (exact mode)")
        return df[m].iloc[0]
    idx = (df["target_recall"].astype(float) - target).abs().idxmin()
    return df.loc[idx]


def _decode_assignment(row: pd.Series, space) -> Dict[str, object]:
    raw = np.array([float(row[k]) for k in space.knob_names], dtype=np.float32)
    return space.raw_numpy_row_to_value_dict(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-config", type=Path, required=True)
    ap.add_argument("--recommend-csv", type=Path, required=True, help="CSV from generate_recommended_configurations_vdtuner_prior.py")
    ap.add_argument("--benchmark-dataset", type=str, required=True, help="datasets.json key for run_engine_test.sh")
    ap.add_argument("--target-recalls", type=str, default="0.86,0.90,0.95")
    ap.add_argument("--exact-target-match", action="store_true", help="Require exact target_recall rows in CSV")
    ap.add_argument("--engine", type=str, default="milvus-single-node")
    ap.add_argument("--process-tag", type=str, default="milvus-p10")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--out-csv", type=Path, default=HERE / "deploy_benchmark_results.csv")
    ap.add_argument("--knob-json", type=Path, default=None)
    args = ap.parse_args()

    if not args.prior_config.is_file():
        raise SystemExit(f"Missing --prior-config: {args.prior_config}")
    if not args.recommend_csv.is_file():
        raise SystemExit(f"Missing --recommend-csv: {args.recommend_csv}")

    target_recalls = _parse_target_recalls(args.target_recalls)
    space = load_prior_space(
        args.knob_json if args.knob_json else None,
        prior_config_path=args.prior_config,
    )
    df = pd.read_csv(args.recommend_csv)
    missing_cols = [k for k in space.knob_names if k not in df.columns]
    if missing_cols:
        raise SystemExit(f"recommend CSV missing knob columns: {missing_cols}")

    results: List[Dict[str, object]] = []
    for trg in target_recalls:
        row = _select_row(df, trg, args.exact_target_match)
        picked = float(row["target_recall"])
        assignment = _decode_assignment(row, space)
        full_conf = space.build_full_milvus_config(assignment)

        idx_copy = dict(full_conf)
        configure_index(*filter_index_rule(idx_copy))
        system_conf = {k: full_conf[k] for k in full_conf if "*" in k}
        configure_system(filter_system_rule(system_conf))

        recall, rps, p95 = run_milvus_benchmark(
            args.engine,
            args.process_tag,
            args.benchmark_dataset,
            use_sudo=not args.no_sudo,
            timeout_sec=args.timeout_sec,
        )
        one = {
            "requested_target_recall": trg,
            "picked_target_recall": picked,
            "measured_recall": float(recall),
            "measured_rps": float(rps),
            "measured_p95_latency": float(p95),
            "index_type": full_conf.get("index_type"),
            "index_build_params": json.dumps(filter_index_rule(dict(full_conf))[1], ensure_ascii=False),
            "search_params": json.dumps(filter_index_rule(dict(full_conf))[2], ensure_ascii=False),
            "system_overrides": json.dumps(system_conf, ensure_ascii=False),
            "resolved_assignment": json.dumps(assignment, ensure_ascii=False),
        }
        results.append(one)
        print(
            f"[done] request={trg:.2f}, picked={picked:.2f}, "
            f"recall={recall:.6f}, rps={rps:.6f}, p95={p95:.6f}"
        )

    out = pd.DataFrame(results)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved benchmark results to: {args.out_csv}")


if __name__ == "__main__":
    main()
