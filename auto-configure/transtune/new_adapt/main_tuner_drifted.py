#!/usr/bin/env python3
"""Tune Milvus on an existing drifted dataset with POBO.

This script follows the same high-level workflow as `main_tuner.py`:
1) Build a real benchmark environment.
2) Run initial sampling.
3) Run iterative tuning.

Difference:
- Benchmark target is an explicit drifted dataset directory (`--dataset-path`),
  so it can tune directly on `.../drifted_full`.
"""

from __future__ import annotations

import json
import signal
import subprocess as sp
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Resolve project paths first, then inject imports for reused tuner modules.
THIS_FILE = Path(__file__).resolve()
NEW_ADAPT_DIR = THIS_FILE.parent
VDTUNER_DIR = NEW_ADAPT_DIR.parent
AUTO_CONFIGURE_DIR = VDTUNER_DIR.parent
PROJECT_ROOT = AUTO_CONFIGURE_DIR.parent
BENCHMARK_ROOT = PROJECT_ROOT / "vector-db-benchmark-master"

sys.path.insert(0, str(AUTO_CONFIGURE_DIR))
sys.path.insert(0, str(VDTUNER_DIR))

from configure import configure_index, configure_system, filter_index_rule, filter_system_rule
from optimizer_pobo_sa import PollingBayesianOptimization
from utils import KnobStand


VDB_TUNING_ROOT = str(PROJECT_ROOT)
KNOB_PATH = f"{VDB_TUNING_ROOT}/auto-configure/whole_param.json"
RUN_CUSTOM_BENCHMARK = NEW_ADAPT_DIR / "run_custom_benchmark.py"

# ============================================
# Configuration: tune this drifted dataset
# ============================================
# =======glove-100-angular dataset
# DATASET_PATH = (
#     BENCHMARK_ROOT
#     / "datasets"
#     / "new_adapt"
#     / "drift-adapt-bo-20260604-184839"
#     / "drifted_full"
# )
# DATASET_NAME = "new-adapt-drift-adapt-bo-20260604-184839-drifted-full"
# VECTOR_SIZE = 100
# DISTANCE = "cosine"
# ENGINE_NAME = "milvus-p10"
# HOST = "127.0.0.1"
# SEED = 1
# ITERATIONS = 100

# Logs and benchmark meta outputs for this tuner run.
# RECORD_LOG_PATH = NEW_ADAPT_DIR / "record_drifted_tuner.log"
# META_OUTPUT_DIR = NEW_ADAPT_DIR / "drifted_tuner_meta"


# ==========arxiv dataset
DATASET_PATH = (
    BENCHMARK_ROOT
    / "datasets"
    / "new_adapt"
    / "drift-adapt-bo-20260613-204055"
    / "drifted_full"
)
DATASET_NAME = "new-adapt-drift-adapt-bo-20260613-204055-drifted-full"
VECTOR_SIZE = 384
DISTANCE = "cosine"
ENGINE_NAME = "milvus-p10"
HOST = "127.0.0.1"
SEED = 1
ITERATIONS = 100
# Logs and benchmark meta outputs for this tuner run.
RECORD_LOG_PATH = NEW_ADAPT_DIR / "record_drifted_tuner_arxiv.log"
META_OUTPUT_DIR = NEW_ADAPT_DIR / "drifted_tuner_meta_arxiv"

# Precision objective shaping:
# - For optimization, precision above TARGET no longer gets extra reward.
# - Precision below FLOOR gets a linear penalty so candidates under the floor
#   become less attractive.
# This encourages higher RPS once recall is around the target band.
USE_TARGETED_PRECISION_OBJECTIVE = True
PRECISION_TARGET = 0.95
PRECISION_FLOOR = 0.90


class DriftedRealEnv:
    """Real benchmark environment over an explicit dataset path."""

    def __init__(
        self,
        dataset_path: Path,
        dataset_name: str,
        vector_size: int,
        distance: str,
        engine_name: str = ENGINE_NAME,
        host: str = HOST,
        knob_path: str = KNOB_PATH,
        record_log_path: Optional[Path] = None,
        meta_output_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset_name = str(dataset_name)
        self.vector_size = int(vector_size)
        self.distance = str(distance)
        self.engine_name = str(engine_name)
        self.host = str(host)

        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())

        self.record_log_path = record_log_path or RECORD_LOG_PATH
        self.meta_output_dir = meta_output_dir or META_OUTPUT_DIR
        self.meta_output_dir.mkdir(parents=True, exist_ok=True)

        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.Y1_record: list[float] = []
        self.Y2_record: list[float] = []
        self.Y4_record: list[float] = []
        self._uploaded_once = False

    @staticmethod
    def _shape_precision_for_optimization(raw_precision: float) -> float:
        """Transform raw precision into an optimization score."""
        if not USE_TARGETED_PRECISION_OBJECTIVE:
            return raw_precision

        if raw_precision >= PRECISION_TARGET:
            # Above target, recall improvements do not increase objective.
            return PRECISION_TARGET

        if raw_precision >= PRECISION_FLOOR:
            # In the acceptable band, keep monotonic preference.
            return raw_precision

        # Below floor, apply linear penalty to discourage low-recall points.
        return raw_precision - (PRECISION_FLOOR - raw_precision)

    def _run_custom_benchmark(
        self,
        iteration: int,
        skip_upload: bool = False,
        rebuild_index: bool = False,
    ) -> tuple[float, float, float, str]:
        """Run one benchmark and return (precision, p95_time, rps, result_file)."""
        result_meta_path = self.meta_output_dir / f"iter-{iteration:04d}.json"
        cmd = [
            "python3",
            str(RUN_CUSTOM_BENCHMARK),
            "--benchmark-root",
            str(BENCHMARK_ROOT),
            "--engine-name",
            self.engine_name,
            "--dataset-name",
            self.dataset_name,
            "--dataset-path",
            str(self.dataset_path),
            "--vector-size",
            str(self.vector_size),
            "--distance",
            self.distance,
            "--host",
            self.host,
            "--result-json",
            str(result_meta_path),
        ]
        if skip_upload:
            cmd.append("--skip-upload")
        if rebuild_index:
            cmd.append("--rebuild-index")

        print(
            f"[iter-{iteration:04d}] benchmark mode: skip_upload={skip_upload}, rebuild_index={rebuild_index}",
            flush=True,
        )
        completed = sp.run(
            cmd,
            cwd=str(NEW_ADAPT_DIR),
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            check=False,
        )
        if completed.stdout:
            sys.stdout.write(completed.stdout)
            sys.stdout.flush()
        if completed.returncode != 0:
            raise RuntimeError(f"run_custom_benchmark failed with return code {completed.returncode}.")

        if not result_meta_path.exists():
            raise FileNotFoundError(f"result meta not found: {result_meta_path}")
        meta = json.loads(result_meta_path.read_text(encoding="utf-8"))
        result_file = Path(meta["result_file"]).resolve()
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        results = payload.get("results", {})

        precision = float(results["mean_precisions"])
        rps = float(results["rps"])
        p95_time = float(results["p95_time"])
        return precision, p95_time, rps, str(result_file)

    def get_state(self, knob_vals_arr: np.ndarray) -> np.ndarray:
        # Return metrics in the same order expected by PollingBayesianOptimization:
        # - col0: precision (bigger is better)
        # - col1: rps (bigger is better)
        # - col2: elapsed seconds for bookkeeping
        Y1, Y2, Y3, Y4 = [], [], [], []

        for record in knob_vals_arr:
            conf_value = [
                self.knob_stand.scale_back(self.names[j], knob_val)[1]
                for j, knob_val in enumerate(record)
            ]
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]
            index_conf = dict(zip(index_name, index_value))
            system_conf = dict(zip(system_name, system_value))

            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            precision, p95_time, rps = 0.1, 0.1, 0.1
            result_file = ""
            skip_upload = self._uploaded_once
            rebuild_index = self._uploaded_once
            try:
                precision, p95_time, rps, result_file = self._run_custom_benchmark(
                    self.sampled_times + 1,
                    skip_upload=skip_upload,
                    rebuild_index=rebuild_index,
                )
                if not skip_upload:
                    self._uploaded_once = True
                self.Y1_record.append(precision)
                self.Y2_record.append(p95_time)
                self.Y4_record.append(rps)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Benchmark execution failed: {exc}")
                traceback.print_exc()

            elapsed_since_last = int(time.time() - self.t2)
            self.sampled_times += 1
            self.t2 = time.time()
            print(f"[{self.sampled_times}] {int(self.t2 - self.t1)} {precision} {p95_time} {elapsed_since_last}")

            self.record_log_path.parent.mkdir(parents=True, exist_ok=True)
            precision_for_opt = self._shape_precision_for_optimization(precision)
            log_entry: dict[str, Any] = {
                "iteration": self.sampled_times,
                "time": int(self.t2 - self.t1),
                "index_conf": index_conf,
                "system_conf": system_conf,
                "precisions": precision,
                "precision_for_opt": precision_for_opt,
                "p95time": p95_time,
                "Time": elapsed_since_last,
                "RPS": rps,
                "result_file": result_file,
                "skip_upload": skip_upload,
                "rebuild_index": rebuild_index,
            }
            with self.record_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            Y1.append(precision_for_opt)
            Y2.append(p95_time)
            Y3.append(elapsed_since_last)
            Y4.append(rps)

        # Optimizer objective keeps (precision, rps). p95_time is retained in logs.
        return np.array([Y1, Y4, Y3]).T

    def default_conf(self) -> list[float]:
        return [
            self.knob_stand.scale_forward(knob_name, knob_info["default"])
            for knob_name, knob_info in self.knob_stand.knobs_detail.items()
        ]


def _register_signal_handlers() -> None:
    def _handle_signal(signum: int, _frame: Any) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
        raise SystemExit(128 + int(signum))

    # Keep nohup behavior.
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            pass


if __name__ == "__main__":
    _register_signal_handlers()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    env = DriftedRealEnv(
        dataset_path=DATASET_PATH,
        dataset_name=DATASET_NAME,
        vector_size=VECTOR_SIZE,
        distance=DISTANCE,
        engine_name=ENGINE_NAME,
        host=HOST,
    )
    model = PollingBayesianOptimization(env, seed=SEED)

    model.init_sample()
    for _ in range(ITERATIONS):
        model.step()
