# -*- coding: utf-8 -*-
"""
End-to-end PGTuner (HNSW-only) on Milvus + vdtuner benchmark wiring.

Phases
------
1. (optional) ``collect`` — grid search via ``run_engine_test.sh``, append ``Data/index_performance_milvus.csv``
2. ``merge`` — join ``data_feature.csv`` + Milvus perf + ``milvus_dataset_sizes.csv`` → ``train_data.csv``
3. ``train-qpp`` — ``query_performance_predict/train.py``
4. ``sync-qpp`` — copy QPP checkpoint + scaler to paths expected by PCR evaluate
5. (optional) ``mirror-test`` — copy train CSV to ``Data/test_data_main.csv`` (single-dataset smoke)
6. ``train-pcr`` — ``parameter_configuration_recommend/train.py`` (TD3 offline training)
7. ``recommend`` — ``evaluate.py`` + ``generate_recommended_configurations.py`` → CSV with
   (M, efConstruction, efSearch) per target recall

vdtuner alignment
-----------------
- ``hnsw_vdtuner_env.HNSWVDTunerEnv`` exposes the same ``get_state`` / ``default_conf`` contract as
  ``auto-configure/transtune`` ``RealEnv`` for HNSW triplets only (see module docstring).

Example::

  # 1) Copy Data/milvus_dataset_sizes.example.csv → Data/milvus_dataset_sizes.csv and edit N, q_queries.

  python run_pgtuner_milvus_pipeline.py all \\
    --pgtuner-filename tiny_1_1_384 \\
    --pgtuner-dataset-key tiny \\
    --benchmark-dataset random-match-int-2048-angular-no-filters \\
    --baseline-efsearch 100 \\
    --skip-collect \\
    --milvus-perf Data/index_performance_milvus.csv

  # With collect (slow): drop --skip-collect and --milvus-perf; uses tiny grid by default.

Extra QPP/PCR flags pass through: ``--dipredict-n-epochs 500 --test-epoches 5 --max-steps 50`` etc.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VDIR = Path(__file__).resolve().parent
QPP = ROOT / "query_performance_predict"
PCR = ROOT / "parameter_configuration_recommend"
DATA = ROOT / "Data"


def _run(cmd: list, cwd: Path) -> None:
    print(f"[pipeline] cd {cwd} && {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="PGTuner Milvus full pipeline")
    ap.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=(
            "collect",
            "merge",
            "train-qpp",
            "sync-qpp",
            "mirror-test",
            "train-pcr",
            "recommend",
            "all",
        ),
    )
    ap.add_argument("--pgtuner-filename", default="tiny_1_1_384", help="FileName in data_feature.csv")
    ap.add_argument("--pgtuner-dataset-key", default="tiny", help="short name for evaluate/generate (filename_dic key)")
    ap.add_argument("--benchmark-dataset", default="", help="vector-db benchmark dataset (required for collect)")
    ap.add_argument("--experiment-mode", default="main")
    ap.add_argument("--baseline-efsearch", type=int, default=100, help="Milvus min ef and CSV baseline row")
    ap.add_argument("--grid", choices=("tiny", "medium", "large"), default="tiny")
    ap.add_argument("--milvus-perf", type=Path, default=DATA / "index_performance_milvus.csv")
    ap.add_argument("--sizes-csv", type=Path, default=DATA / "milvus_dataset_sizes.csv")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-train-qpp", action="store_true")
    ap.add_argument("--skip-sync-qpp", action="store_true")
    ap.add_argument("--skip-mirror-test", action="store_true")
    ap.add_argument("--skip-train-pcr", action="store_true")
    ap.add_argument("--skip-recommend", action="store_true")
    ap.add_argument("--overwrite-train-data", action="store_true", help="Required for merge→train_data.csv replace")
    ap.add_argument("--engine", default="milvus-single-node")
    ap.add_argument("--process-tag", default="milvus-p10")
    ap.add_argument("--no-sudo-benchmark", action="store_true")

    args, passthrough = ap.parse_known_args()

    sizes_example = DATA / "milvus_dataset_sizes.example.csv"
    if not args.sizes_csv.is_file():
        if sizes_example.is_file():
            shutil.copy2(sizes_example, args.sizes_csv)
            print(f"Copied {sizes_example} -> {args.sizes_csv}; edit N and q_queries.", flush=True)
        else:
            sys.exit(f"Missing {args.sizes_csv}")

    phases = []
    if args.phase == "all":
        phases = ["collect", "merge", "train-qpp", "sync-qpp", "mirror-test", "train-pcr", "recommend"]
    else:
        phases = [args.phase]

    for ph in phases:
        if ph == "collect":
            if args.skip_collect:
                continue
            if not args.benchmark_dataset:
                sys.exit("collect needs --benchmark-dataset")
            cmd = [
                sys.executable,
                str(VDIR / "milvus_qpp_data.py"),
                "collect",
                "--pgtuner-filename",
                args.pgtuner_filename,
                "--benchmark-dataset",
                args.benchmark_dataset,
                "--sizes-csv",
                str(args.sizes_csv),
                "--out-csv",
                str(args.milvus_perf),
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

        elif ph == "merge":
            if args.skip_merge:
                continue
            if not args.overwrite_train_data:
                sys.exit("merge writes Data/train_data.csv; pass --overwrite-train-data to confirm")
            if not args.milvus_perf.is_file():
                sys.exit(f"Missing Milvus perf CSV: {args.milvus_perf}")
            cmd = [
                sys.executable,
                str(VDIR / "milvus_qpp_data.py"),
                "merge",
                "--milvus-perf",
                str(args.milvus_perf),
                "--sizes-csv",
                str(args.sizes_csv),
                "--out",
                str(DATA / "train_data.csv"),
            ]
            _run(cmd, VDIR)

        elif ph == "train-qpp":
            if args.skip_train_qpp:
                continue
            cmd = [sys.executable, "train.py"] + passthrough
            _run(cmd, QPP)

        elif ph == "sync-qpp":
            if args.skip_sync_qpp:
                continue
            sys.path.insert(0, str(VDIR))
            from pgtuner_paths import sync_qpp_artifacts_for_pcr_evaluate  # noqa: E402

            # Defaults must match query_performance_predict/Args.py unless user overrode on CLI (not parsed here).
            dipredict_layer_sizes = "[14, 128, 256, 64, 3]"
            dipredict_n_epochs = 3000
            dipredict_batch_size = 4096
            dipredict_lr = 0.001
            max_selected_num = 14
            i = 0
            pt = list(passthrough)
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

            sync_qpp_artifacts_for_pcr_evaluate(
                args.experiment_mode,
                args.pgtuner_dataset_key,
                dipredict_layer_sizes=dipredict_layer_sizes,
                dipredict_n_epochs=dipredict_n_epochs,
                dipredict_batch_size=dipredict_batch_size,
                dipredict_lr=dipredict_lr,
                max_selected_num=max_selected_num,
            )
            print("[pipeline] QPP artifacts synced for PCR evaluate.", flush=True)

        elif ph == "mirror-test":
            if args.skip_mirror_test:
                continue
            train_csv = DATA / "train_data.csv"
            if not train_csv.is_file():
                sys.exit(f"Missing {train_csv}")
            shutil.copy2(train_csv, DATA / "test_data_main.csv")
            print("[pipeline] Copied train_data.csv -> test_data_main.csv", flush=True)

        elif ph == "train-pcr":
            if args.skip_train_pcr:
                continue
            cmd = [
                sys.executable,
                "train.py",
                "--baseline-efsearch",
                str(args.baseline_efsearch),
            ] + passthrough
            _run(cmd, PCR)

        elif ph == "recommend":
            if args.skip_recommend:
                continue
            ev = [
                sys.executable,
                "evaluate.py",
                "--dataset-name",
                args.pgtuner_dataset_key,
                "--experiment-mode",
                args.experiment_mode,
                "--baseline-efsearch",
                str(args.baseline_efsearch),
            ] + passthrough
            _run(ev, PCR)
            gen = [
                sys.executable,
                "generate_recommended_configurations.py",
                "--dataset-name",
                args.pgtuner_dataset_key,
                "--experiment-mode",
                args.experiment_mode,
            ] + passthrough
            _run(gen, PCR)


if __name__ == "__main__":
    main()
