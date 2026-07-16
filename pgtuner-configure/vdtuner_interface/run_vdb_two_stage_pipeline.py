# -*- coding: utf-8 -*-
"""
Two-stage pipeline aligned with the PGTuner paper:

1) Pre-training on dataset A:
   extract A features -> dense collect on A -> merge -> train QPP(A, fixed scaler) -> train PCR base policy.
2) Online transfer on dataset B:
   extract B features -> DSD similarity check -> collect only baseline (+ optional small active batches) ->
   transfer QPP on A+B(small) using fixed A scaler -> sync artifacts -> online fine-tune/evaluate PCR -> generate.

Compared with the old entrypoint, this avoids:
- full-grid collection on B (data leakage / over-collecting)
- training PCR from scratch on B
- replacing QPP scaler with B-only scaler.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VDIR = Path(__file__).resolve().parent
QPP = ROOT / "query_performance_predict"
PCR = ROOT / "parameter_configuration_recommend"
DATA = ROOT / "Data"

sys.path.insert(0, str(VDIR))
from vdb.registry import vdb_file_name  # noqa: E402


def _pipeline_log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [vdb-pipeline] {msg}", flush=True)


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


def _hnsw_grid(grid: str) -> Tuple[List[int], List[int], List[int]]:
    if grid == "tiny":
        return [20, 40, 80, 120, 200], [4, 8, 16, 32], [100, 200, 400, 600]
    if grid == "medium":
        return [20, 40, 60, 100, 160, 240, 360], [4, 8, 12, 16, 24, 32], [100, 200, 350, 500, 700, 900]
    return (
        [20, 40, 60, 80, 100, 140, 180, 220, 260, 300],
        [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64],
        [100, 150, 200, 300, 400, 500, 600, 800, 1000],
    )


def _all_hnsw_triplets(grid: str) -> List[Tuple[int, int, int]]:
    efcs, ms, efs = _hnsw_grid(grid)
    out: List[Tuple[int, int, int]] = []
    for efc in efcs:
        for m in ms:
            if m > efc:
                continue
            for ef in efs:
                out.append((efc, m, ef))
    return out


def _triplet_matrix(items: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    arr = np.array(items, dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    scale = np.where(maxs - mins > 0, maxs - mins, 1.0)
    return (arr - mins) / scale


def _select_diverse_triplets(
    candidates: Sequence[Tuple[int, int, int]],
    already_selected: Sequence[Tuple[int, int, int]],
    k: int,
    seed: int,
) -> List[Tuple[int, int, int]]:
    if k <= 0 or not candidates:
        return []
    cand = list(candidates)
    cand_norm = _triplet_matrix(cand)
    chosen: List[Tuple[int, int, int]] = []
    rng = np.random.default_rng(seed)

    if already_selected:
        ref_norm = _triplet_matrix(list(already_selected))
        min_dist = np.linalg.norm(cand_norm[:, None, :] - ref_norm[None, :, :], axis=2).min(axis=1)
        first_idx = int(np.argmax(min_dist))
    else:
        first_idx = int(rng.integers(0, len(cand)))
    chosen.append(cand[first_idx])

    while len(chosen) < k:
        remain = [x for x in cand if x not in chosen]
        if not remain:
            break
        ref = list(already_selected) + chosen
        remain_norm = _triplet_matrix(remain)
        ref_norm = _triplet_matrix(ref)
        min_dist = np.linalg.norm(remain_norm[:, None, :] - ref_norm[None, :, :], axis=2).min(axis=1)
        idx = int(np.argmax(min_dist))
        chosen.append(remain[idx])
    return chosen


def _collect_hnsw_rows(
    dataset_file_name: str,
    benchmark_dataset: str,
    triplets: Iterable[Tuple[int, int, int]],
    out_csv: Path,
    engine: str,
    process_tag: str,
    use_sudo: bool,
    timeout_sec: int = 2000,
    resume: bool = True,
) -> None:
    from benchmark_runner import run_milvus_benchmark
    from hnsw_vdtuner_env import apply_hnsw_triplet

    sizes = pd.read_csv(DATA / "milvus_dataset_sizes.csv")
    row = sizes[sizes["FileName"] == dataset_file_name]
    if len(row) == 0:
        raise SystemExit(f"Missing {dataset_file_name} in {DATA / 'milvus_dataset_sizes.csv'}")
    n_base = float(row.iloc[0]["N"])
    q_queries = float(row.iloc[0]["q_queries"])

    header = [
        "FileName",
        "efConstruction",
        "M",
        "efSearch",
        "construction_time",
        "memory",
        "recall",
        "search_time",
        "construct_dc_counts",
        "search_dc_counts",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_csv.is_file():
        with out_csv.open("w", encoding="utf-8", newline="") as fp:
            csv.writer(fp).writerow(header)

    exist = set()
    if resume and out_csv.is_file():
        old = pd.read_csv(out_csv)
        if len(old):
            exist = set(zip(old["FileName"], old["efConstruction"], old["M"], old["efSearch"]))

    for efc, m, ef in triplets:
        key = (dataset_file_name, int(efc), int(m), int(ef))
        if key in exist:
            continue
        apply_hnsw_triplet(int(m), int(efc), int(ef))
        prec, _rps, p95 = run_milvus_benchmark(
            engine,
            process_tag,
            benchmark_dataset,
            use_sudo=use_sudo,
            timeout_sec=timeout_sec,
        )
        search_time = float(p95)
        recall = float(prec)
        construct_dc = max(1, int(int(m) * int(efc) * math.sqrt(max(n_base, 1.0))))
        search_dc = max(1, int(search_time * 1e6 * max(q_queries, 1.0)))
        with out_csv.open("a", encoding="utf-8", newline="") as fp:
            csv.writer(fp).writerow(
                [dataset_file_name, int(efc), int(m), int(ef), 0.0, 0, recall, search_time, construct_dc, search_dc]
            )
        _pipeline_log(f"active-collect logged {key} recall={recall:.4f} p95={search_time:.4f}")


def _feature_distance(a_tag: str, b_tag: str) -> float:
    feat_csv = DATA / "vdb_data_feature.csv"
    if not feat_csv.is_file():
        raise SystemExit(f"Missing feature table: {feat_csv}")
    df = pd.read_csv(feat_csv)
    if "FileName" not in df.columns:
        raise SystemExit(f"Invalid feature table (missing FileName): {feat_csv}")
    num_cols = [c for c in df.columns if c != "FileName"]
    if not num_cols:
        raise SystemExit(f"Invalid feature table (no numeric feature columns): {feat_csv}")
    dfn = df[num_cols].astype(float)
    mu = dfn.mean(axis=0)
    sigma = dfn.std(axis=0).replace(0.0, 1.0)
    a = df[df["FileName"] == a_tag]
    b = df[df["FileName"] == b_tag]
    if len(a) == 0 or len(b) == 0:
        raise SystemExit(f"Cannot run similarity detect: missing rows for {a_tag} or {b_tag} in {feat_csv}")
    za = (a.iloc[0][num_cols].astype(float) - mu) / sigma
    zb = (b.iloc[0][num_cols].astype(float) - mu) / sigma
    dist = float(np.linalg.norm(za.values - zb.values) / np.sqrt(len(num_cols)))
    return dist


def _merge_to_train(milvus_perf: Path, out_train_csv: Path, file_name: str) -> None:
    _run(
        [
            sys.executable,
            str(VDIR / "milvus_qpp_data.py"),
            "merge",
            "--milvus-perf",
            str(milvus_perf),
            "--out",
            str(out_train_csv),
            "--file-name",
            file_name,
        ],
        VDIR,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="all", choices=["all", "pretrain", "online"])
    ap.add_argument("--qpp-dataset", required=True, help="datasets.json name for pre-training dataset A")
    ap.add_argument("--pcr-dataset", required=True, help="datasets.json name for online tuning dataset B")
    ap.add_argument("--experiment-mode", default="main")
    ap.add_argument("--baseline-efsearch", type=int, default=100)
    ap.add_argument("--overwrite-train-data", action="store_true")
    ap.add_argument("--grid", choices=("tiny", "medium", "large"), default="tiny", help="dense collect grid on A")
    ap.add_argument(
        "--active-grid",
        choices=("tiny", "medium", "large"),
        default="tiny",
        help="candidate grid for B active collection",
    )
    ap.add_argument("--active-rounds", type=int, default=7, help="active rounds on B")
    ap.add_argument("--active-per-round", type=int, default=2, help="points collected per active round")
    ap.add_argument("--dsd-threshold", type=float, default=1.0, help="B is similar to A if distance <= threshold")
    ap.add_argument("--force-active-collect", action="store_true")
    ap.add_argument("--engine", default="milvus-single-node")
    ap.add_argument("--process-tag", default="milvus-p10")
    ap.add_argument("--no-sudo-benchmark", action="store_true")
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
    ap.add_argument(
        "--reset-qpp-scaler-before-base-train",
        action="store_true",
        help="remove query_performance_predict/scaler_paras/feature_standard.npz before A training",
    )

    args, passthrough = ap.parse_known_args()
    sys.path.insert(0, str(VDIR))

    qpp_tag = vdb_file_name(args.qpp_dataset)
    pcr_tag = vdb_file_name(args.pcr_dataset)
    if qpp_tag == pcr_tag:
        sys.exit("--qpp-dataset and --pcr-dataset must differ")
    if not args.overwrite_train_data:
        sys.exit("This entrypoint requires --overwrite-train-data to avoid silently reusing stale train_data.csv")

    idx_qpp = DATA / "index_performance_qpp.csv"
    train_qpp = DATA / "train_data_qpp.csv"
    idx_target = DATA / "index_performance_target_active.csv"
    train_target = DATA / "train_data_target_active.csv"
    train_transfer = DATA / "train_data_qpp_transfer.csv"
    dip = _parse_dipredict(passthrough)

    def phase_extract_qpp() -> None:
        if args.skip_extract_qpp:
            return
        _run([sys.executable, str(VDIR / "extract_vdb_features.py"), "--dataset", args.qpp_dataset], VDIR)

    def phase_collect_qpp() -> None:
        if args.skip_collect_qpp:
            return
        cmd = [
            sys.executable,
            str(VDIR / "milvus_qpp_data.py"),
            "collect",
            "--pgtuner-filename",
            qpp_tag,
            "--benchmark-dataset",
            args.qpp_dataset,
            "--out-csv",
            str(idx_qpp),
            "--grid",
            args.grid,
            "--engine",
            args.engine,
            "--process-tag",
            args.process_tag,
            "--resume",
        ]
        if args.no_sudo_benchmark:
            cmd.append("--no-sudo")
        _run(cmd, VDIR)

    def phase_merge_qpp() -> None:
        if args.skip_merge_qpp:
            return
        _merge_to_train(idx_qpp, train_qpp, qpp_tag)

    def phase_train_qpp_base() -> None:
        if args.skip_train_qpp:
            return
        scaler = QPP / "scaler_paras" / "feature_standard.npz"
        if args.reset_qpp_scaler_before_base_train and scaler.is_file():
            scaler.unlink()
            _pipeline_log(f"removed {scaler}")
        shutil.copy2(train_qpp, DATA / "train_data.csv")
        _run([sys.executable, "train.py"] + passthrough, QPP)

    def phase_train_pcr_base() -> None:
        if args.skip_train_pcr_base:
            return
        shutil.copy2(train_qpp, DATA / "train_data.csv")
        _run(
            [sys.executable, "train.py", "--baseline-efsearch", str(args.baseline_efsearch)] + passthrough,
            PCR,
        )

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
        # Always collect baseline row so evaluate.py has a real default performance anchor on dataset B.
        _collect_hnsw_rows(
            dataset_file_name=pcr_tag,
            benchmark_dataset=args.pcr_dataset,
            triplets=[(20, 4, int(args.baseline_efsearch))],
            out_csv=idx_target,
            engine=args.engine,
            process_tag=args.process_tag,
            use_sudo=not args.no_sudo_benchmark,
            resume=True,
        )
        need_active = args.force_active_collect or (not similar_flag)
        if not need_active:
            _pipeline_log("DSD says similar; skip extra active collection and keep baseline-only online data.")
            return

        candidates = [x for x in _all_hnsw_triplets(args.active_grid) if x != (20, 4, int(args.baseline_efsearch))]
        selected: List[Tuple[int, int, int]] = [(20, 4, int(args.baseline_efsearch))]
        for rid in range(args.active_rounds):
            remain = [x for x in candidates if x not in selected]
            picks = _select_diverse_triplets(remain, selected, args.active_per_round, seed=42 + rid)
            if not picks:
                break
            _pipeline_log(f"active round {rid + 1}/{args.active_rounds}, collecting {len(picks)} points")
            _collect_hnsw_rows(
                dataset_file_name=pcr_tag,
                benchmark_dataset=args.pcr_dataset,
                triplets=picks,
                out_csv=idx_target,
                engine=args.engine,
                process_tag=args.process_tag,
                use_sudo=not args.no_sudo_benchmark,
                resume=True,
            )
            selected.extend(picks)

    def phase_transfer_qpp(similar_flag: bool) -> None:
        if args.skip_transfer_qpp:
            return
        if not idx_target.is_file():
            raise SystemExit(f"Missing target collection CSV: {idx_target}")
        _merge_to_train(idx_target, train_target, pcr_tag)
        need_transfer = args.force_active_collect or (not similar_flag)
        if not need_transfer:
            _pipeline_log("skip QPP transfer: using base QPP directly for similar dataset.")
            shutil.copy2(train_target, DATA / "test_data_main.csv")
            return

        base_df = pd.read_csv(train_qpp)
        target_df = pd.read_csv(train_target)
        merged = pd.concat([base_df, target_df], axis=0, ignore_index=True)
        key_cols = [c for c in ("FileName", "efConstruction", "M", "efSearch") if c in merged.columns]
        if key_cols:
            merged = merged.drop_duplicates(subset=key_cols, keep="last")
        merged.to_csv(train_transfer, index=False)
        shutil.copy2(train_transfer, DATA / "train_data.csv")
        _run([sys.executable, "train.py"] + passthrough, QPP)
        shutil.copy2(train_target, DATA / "test_data_main.csv")

    def phase_sync_eval() -> None:
        if args.skip_sync_eval:
            return
        from pgtuner_paths import sync_qpp_artifacts_for_pcr_evaluate  # noqa: E402

        sync_qpp_artifacts_for_pcr_evaluate(
            args.experiment_mode,
            pcr_tag,
            dipredict_layer_sizes=dip["dipredict_layer_sizes"],
            dipredict_n_epochs=dip["dipredict_n_epochs"],
            dipredict_batch_size=dip["dipredict_batch_size"],
            dipredict_lr=dip["dipredict_lr"],
            max_selected_num=dip["max_selected_num"],
        )

    def phase_evaluate() -> None:
        if args.skip_evaluate:
            return
        _run(
            [
                sys.executable,
                "evaluate.py",
                "--dataset-name",
                pcr_tag,
                "--experiment-mode",
                args.experiment_mode,
                "--baseline-efsearch",
                str(args.baseline_efsearch),
            ]
            + passthrough,
            PCR,
        )

    def phase_generate() -> None:
        if args.skip_generate:
            return
        _run(
            [
                sys.executable,
                "generate_recommended_configurations.py",
                "--dataset-name",
                pcr_tag,
                "--experiment-mode",
                args.experiment_mode,
            ]
            + passthrough,
            PCR,
        )

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
