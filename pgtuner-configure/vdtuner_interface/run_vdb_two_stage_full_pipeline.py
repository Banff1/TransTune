# -*- coding: utf-8 -*-
"""
Prior/full-knob two-stage pipeline aligned with PGTuner:

1) Pre-training on dataset A:
   extract -> dense prior collect on A -> merge -> train prior-QPP (fixed prior scaler) -> train prior-PCR base policy.
2) Online tuning on dataset B:
   extract -> DSD -> baseline-only collect (+ optional small active batches) -> merge ->
   transfer prior-QPP on A+B(small) using fixed prior scaler -> sync -> evaluate/generate on B.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VDIR = Path(__file__).resolve().parent
QPP = ROOT / "query_performance_predict"
PCR = ROOT / "parameter_configuration_recommend"
DATA = ROOT / "Data"

sys.path.insert(0, str(VDIR))
from pgtuner_paths import (  # noqa: E402
    sync_qpp_transfer_to_pcr_evaluate_prior,
    vdtuner_prior_qpp_meta_path,
)
from vdb.registry import vdb_file_name  # noqa: E402


def _pipeline_log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [vdb-prior-pipeline] {msg}", flush=True)


def _run(cmd: list, cwd: Path) -> None:
    _pipeline_log(f"cd {cwd} && {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _parse_dipredict(pt: list) -> dict:
    dipredict_layer_sizes = "[14, 128, 256, 64, 3]"
    dipredict_n_epochs = 3000
    dipredict_batch_size = 4096
    dipredict_lr = 0.001
    max_selected_num = 14
    i = 0
    while i < len(pt):
        if pt[i] == "--dipredict-layer-sizes" and i + 1 < len(pt):
            dipredict_layer_sizes = pt[i + 1]
            i += 2
        elif pt[i] == "--dipredict-n-epochs" and i + 1 < len(pt):
            dipredict_n_epochs = int(pt[i + 1])
            i += 2
        elif pt[i] == "--dipredict-batch-size" and i + 1 < len(pt):
            dipredict_batch_size = int(pt[i + 1])
            i += 2
        elif pt[i] == "--dipredict-lr" and i + 1 < len(pt):
            dipredict_lr = float(pt[i + 1])
            i += 2
        elif pt[i] == "--max-selected-num" and i + 1 < len(pt):
            max_selected_num = int(pt[i + 1])
            i += 2
        else:
            i += 1
    return {
        "dipredict_layer_sizes": dipredict_layer_sizes,
        "dipredict_n_epochs": dipredict_n_epochs,
        "dipredict_batch_size": dipredict_batch_size,
        "dipredict_lr": dipredict_lr,
        "max_selected_num": max_selected_num,
    }


def _passthrough_with_meta_layer_sizes(passthrough: list) -> list:
    meta_path = vdtuner_prior_qpp_meta_path()
    if not meta_path.is_file():
        return passthrough
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ls = str(meta["dipredict_layer_sizes"])
    out = list(passthrough)
    for i in range(len(out) - 1):
        if out[i] == "--dipredict-layer-sizes":
            return out
    out.extend(["--dipredict-layer-sizes", ls])
    return out


def _feature_distance(a_tag: str, b_tag: str) -> float:
    feat_csv = DATA / "vdb_data_feature.csv"
    if not feat_csv.is_file():
        raise SystemExit(f"Missing feature table: {feat_csv}")
    df = pd.read_csv(feat_csv)
    if "FileName" not in df.columns:
        raise SystemExit(f"Invalid feature table (missing FileName): {feat_csv}")
    num_cols = [c for c in df.columns if c != "FileName"]
    dfn = df[num_cols].astype(float)
    mu = dfn.mean(axis=0)
    sigma = dfn.std(axis=0).replace(0.0, 1.0)
    a = df[df["FileName"] == a_tag]
    b = df[df["FileName"] == b_tag]
    if len(a) == 0 or len(b) == 0:
        raise SystemExit(f"Cannot run similarity detect: missing rows for {a_tag} or {b_tag} in {feat_csv}")
    za = (a.iloc[0][num_cols].astype(float) - mu) / sigma
    zb = (b.iloc[0][num_cols].astype(float) - mu) / sigma
    return float(np.linalg.norm(za.values - zb.values) / np.sqrt(len(num_cols)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="all", choices=["all", "pretrain", "online"])
    ap.add_argument(
        "--prior-config",
        type=Path,
        required=True,
        help="Unified JSON: inline knob specs and/or index_types, tune_knobs, overrides, baseline",
    )
    ap.add_argument("--knob-json", type=Path, default=None, help="Fallback whole_param.json if prior file has no inline knobs")
    ap.add_argument("--qpp-dataset", required=True)
    ap.add_argument("--pcr-dataset", required=True)
    ap.add_argument("--experiment-mode", default="main")
    ap.add_argument("--overwrite-train-data", action="store_true")
    ap.add_argument("--max-configurations", type=int, default=100_000, help="dense collect budget for A")
    ap.add_argument(
        "--sampling",
        choices=("auto", "exact", "lhs"),
        default="auto",
        help="A-collect mode: auto=factorial if fits else LHS; lhs=always LHS; exact=factorial only",
    )
    ap.add_argument("--lhs-seed", type=int, default=42, help="Seed for LHS sampling")
    ap.add_argument("--active-rounds", type=int, default=7, help="active rounds on B")
    ap.add_argument("--active-per-round", type=int, default=2, help="new points per active round on B")
    ap.add_argument("--dsd-threshold", type=float, default=1.0, help="B is similar to A if distance <= threshold")
    ap.add_argument("--force-active-collect", action="store_true")
    ap.add_argument("--engine", default="milvus-single-node")
    ap.add_argument("--process-tag", default="milvus-p10")
    ap.add_argument("--no-sudo-benchmark", action="store_true")
    ap.add_argument("--clean-qpp-scaler", action="store_true", help="Delete feature_standard_vdtuner_prior.npz before A QPP train")
    ap.add_argument("--skip-extract-qpp", action="store_true")
    ap.add_argument("--skip-collect-qpp", action="store_true")
    ap.add_argument("--skip-merge-qpp", action="store_true")
    ap.add_argument("--skip-train-qpp", action="store_true")
    ap.add_argument("--skip-train-pcr-base", action="store_true")
    ap.add_argument("--skip-extract-pcr", action="store_true")
    ap.add_argument("--skip-similarity-detect", action="store_true")
    ap.add_argument("--skip-active-collect", action="store_true")
    ap.add_argument("--skip-transfer-qpp", action="store_true")
    ap.add_argument("--skip-sync-eval", action="store_true")
    ap.add_argument("--skip-evaluate", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")

    args, passthrough = ap.parse_known_args()
    sys.path.insert(0, str(VDIR))

    prior_path = args.prior_config.resolve()
    if not prior_path.is_file():
        sys.exit(f"Missing --prior-config: {prior_path}")
    if not args.overwrite_train_data:
        sys.exit("This entrypoint requires --overwrite-train-data to avoid stale train_data.csv reuse.")

    qpp_tag = vdb_file_name(args.qpp_dataset)
    pcr_tag = vdb_file_name(args.pcr_dataset)
    if qpp_tag == pcr_tag:
        sys.exit("--qpp-dataset and --pcr-dataset must differ")

    idx_qpp = DATA / "index_performance_qpp_prior.csv"
    idx_target = DATA / "index_performance_target_active_prior.csv"
    train_qpp = DATA / "train_data_qpp_prior.csv"
    train_target = DATA / "train_data_target_active_prior.csv"
    train_transfer = DATA / "train_data_qpp_transfer_prior.csv"
    scaler_prior = QPP / "scaler_paras" / "feature_standard_vdtuner_prior.npz"

    dip = _parse_dipredict(passthrough)
    knob_arg: list = []
    if args.knob_json is not None:
        knob_arg = ["--knob-json", str(args.knob_json)]

    def collect_prior_cmd(
        out_csv: Path,
        pgtuner_fn: str,
        bench_ds: str,
        max_cfg: int,
        sampling: str,
        lhs_seed: int,
    ) -> list:
        cmd = [
            sys.executable,
            str(VDIR / "milvus_qpp_data.py"),
            "collect-prior",
            "--prior-config",
            str(prior_path),
            "--pgtuner-filename",
            pgtuner_fn,
            "--benchmark-dataset",
            bench_ds,
            "--out-csv",
            str(out_csv),
            "--max-configurations",
            str(max_cfg),
            "--sampling",
            sampling,
            "--lhs-seed",
            str(lhs_seed),
            "--engine",
            args.engine,
            "--process-tag",
            args.process_tag,
            "--resume",
        ]
        cmd.extend(knob_arg)
        if args.no_sudo_benchmark:
            cmd.append("--no-sudo")
        return cmd

    def phase_extract_qpp() -> None:
        if args.skip_extract_qpp:
            return
        _run([sys.executable, str(VDIR / "extract_vdb_features.py"), "--dataset", args.qpp_dataset], VDIR)

    def phase_collect_qpp() -> None:
        if args.skip_collect_qpp:
            return
        _run(
            collect_prior_cmd(
                out_csv=idx_qpp,
                pgtuner_fn=qpp_tag,
                bench_ds=args.qpp_dataset,
                max_cfg=args.max_configurations,
                sampling=args.sampling,
                lhs_seed=args.lhs_seed,
            ),
            VDIR,
        )

    def phase_merge_qpp() -> None:
        if args.skip_merge_qpp:
            return
        _run(
            [
                sys.executable,
                str(VDIR / "milvus_qpp_data.py"),
                "merge",
                "--milvus-perf",
                str(idx_qpp),
                "--out",
                str(train_qpp),
                "--file-name",
                qpp_tag,
            ],
            VDIR,
        )

    def phase_train_qpp_base() -> None:
        if args.skip_train_qpp:
            return
        if args.clean_qpp_scaler and scaler_prior.is_file():
            scaler_prior.unlink()
            _pipeline_log(f"removed {scaler_prior}")
        shutil.copy2(train_qpp, DATA / "train_data.csv")
        qpp_pass = ["--prior-config", str(prior_path)] + knob_arg + passthrough
        _run([sys.executable, "train_vdtuner_prior.py"] + qpp_pass, QPP)

    def phase_train_pcr_base() -> None:
        if args.skip_train_pcr_base:
            return
        shutil.copy2(train_qpp, DATA / "train_data.csv")
        pcr_pass = (
            _passthrough_with_meta_layer_sizes(passthrough)
            + [
                "--prior-config",
                str(prior_path),
                "--feature-standard-npz",
                str(scaler_prior),
            ]
            + knob_arg
        )
        _run([sys.executable, "train_vdtuner_prior.py"] + pcr_pass, PCR)

    def phase_extract_pcr() -> None:
        if args.skip_extract_pcr:
            return
        _run([sys.executable, str(VDIR / "extract_vdb_features.py"), "--dataset", args.pcr_dataset], VDIR)

    def phase_similarity_detect() -> bool:
        if args.skip_similarity_detect:
            return False
        dist = _feature_distance(qpp_tag, pcr_tag)
        similar = dist <= args.dsd_threshold
        _pipeline_log(f"DSD distance={dist:.6f}, threshold={args.dsd_threshold:.6f}, similar={similar}")
        return similar

    def phase_collect_target_minimal(similar_flag: bool) -> None:
        if args.skip_active_collect:
            return
        # baseline-only collect: collect-prior always includes baseline from prior-config first.
        _run(
            collect_prior_cmd(
                out_csv=idx_target,
                pgtuner_fn=pcr_tag,
                bench_ds=args.pcr_dataset,
                max_cfg=0,
                sampling="lhs",
                lhs_seed=args.lhs_seed,
            ),
            VDIR,
        )
        need_active = args.force_active_collect or (not similar_flag)
        if not need_active:
            _pipeline_log("DSD says similar; keep baseline-only target data.")
            return
        for rid in range(args.active_rounds):
            _pipeline_log(f"active round {rid + 1}/{args.active_rounds}, collect {args.active_per_round} points")
            _run(
                collect_prior_cmd(
                    out_csv=idx_target,
                    pgtuner_fn=pcr_tag,
                    bench_ds=args.pcr_dataset,
                    max_cfg=args.active_per_round,
                    sampling="lhs",
                    lhs_seed=args.lhs_seed + rid + 1,
                ),
                VDIR,
            )

    def phase_transfer_qpp(similar_flag: bool) -> None:
        if args.skip_transfer_qpp:
            return
        _run(
            [
                sys.executable,
                str(VDIR / "milvus_qpp_data.py"),
                "merge",
                "--milvus-perf",
                str(idx_target),
                "--out",
                str(train_target),
                "--file-name",
                pcr_tag,
            ],
            VDIR,
        )
        need_transfer = args.force_active_collect or (not similar_flag)
        if not need_transfer:
            _pipeline_log("skip QPP transfer: using base prior-QPP directly for similar dataset.")
            shutil.copy2(train_target, DATA / "test_data_main.csv")
            return

        base_df = pd.read_csv(train_qpp)
        target_df = pd.read_csv(train_target)
        merged = pd.concat([base_df, target_df], axis=0, ignore_index=True)
        perf_cols = {
            "construction_time",
            "memory",
            "recall",
            "search_time",
            "construct_dc_counts",
            "search_dc_counts",
            "average_construct_dc_counts",
            "average_search_dc_counts",
            "whole_search_time",
        }
        key_cols = [c for c in merged.columns if c not in perf_cols]
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        merged.to_csv(train_transfer, index=False)
        shutil.copy2(train_transfer, DATA / "train_data.csv")
        qpp_pass = ["--prior-config", str(prior_path)] + knob_arg + passthrough
        _run([sys.executable, "train_vdtuner_prior.py"] + qpp_pass, QPP)
        shutil.copy2(train_target, DATA / "test_data_main.csv")

    def phase_sync_eval() -> None:
        if args.skip_sync_eval:
            return
        sync_qpp_transfer_to_pcr_evaluate_prior(
            args.experiment_mode,
            pcr_tag,
            scaler_prior,
            max_selected_num=dip["max_selected_num"],
        )

    def phase_evaluate() -> None:
        if args.skip_evaluate:
            return
        ev_pass = (
            _passthrough_with_meta_layer_sizes(passthrough)
            + [
                "--dataset-name",
                pcr_tag,
                "--experiment-mode",
                args.experiment_mode,
                "--prior-config",
                str(prior_path),
            ]
            + knob_arg
        )
        _run([sys.executable, "evaluate_vdtuner_prior.py"] + ev_pass, PCR)

    def phase_generate() -> None:
        if args.skip_generate:
            return
        gen_pass = (
            _passthrough_with_meta_layer_sizes(passthrough)
            + [
                "--dataset-name",
                pcr_tag,
                "--experiment-mode",
                args.experiment_mode,
                "--prior-config",
                str(prior_path),
            ]
            + knob_arg
        )
        _run([sys.executable, "generate_recommended_configurations_vdtuner_prior.py"] + gen_pass, PCR)

    def run_pretrain() -> None:
        phase_extract_qpp()
        phase_collect_qpp()
        phase_merge_qpp()
        phase_train_qpp_base()
        phase_train_pcr_base()

    def run_online() -> None:
        phase_extract_pcr()
        similar_flag = phase_similarity_detect()
        phase_collect_target_minimal(similar_flag)
        phase_transfer_qpp(similar_flag)
        phase_sync_eval()
        phase_evaluate()
        phase_generate()

    if args.phase == "pretrain":
        run_pretrain()
    elif args.phase == "online":
        run_online()
    elif args.phase == "all":
        run_pretrain()
        run_online()
    else:
        sys.exit("Unsupported phase.")


if __name__ == "__main__":
    main()
