#!/usr/bin/env python3
"""Run main_tuner workflow with per-iteration overhead timing.

Initial sampling runs 7 rounds (one per index type). From round 8 onward,
each step records:
  - bo_propose_s: Bayesian optimization time to propose the next point
  - benchmark_s:  script import + performance test time (env.get_state)
  - bo_update_s:  GP model refit after receiving benchmark results
  - total_s:      full step wall time
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

OVERHEAD_DIR = Path(__file__).resolve().parent
VDTUNER_DIR = OVERHEAD_DIR.parent
AUTO_CONFIGURE_DIR = VDTUNER_DIR.parent
sys.path.insert(0, str(AUTO_CONFIGURE_DIR))
sys.path.insert(0, str(VDTUNER_DIR))
sys.path.insert(0, str(OVERHEAD_DIR))

from timed_pobo import TimedPollingBayesianOptimization, WARMUP_ROUNDS
from utils import RealEnv


def _register_signal_handlers() -> None:
    def _handle_signal(signum: int, _frame) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
        raise SystemExit(128 + int(signum))

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

    # Same defaults as ../main_tuner.py
    DATASET = "arxiv-titles-384-angular-no-filters"
    SEED = 1
    ITERATIONS = 30

    overhead_dir = Path(__file__).resolve().parent
    timing_log = overhead_dir / "overhead_timing3.jsonl"

    print(
        f"[overhead] warmup rounds={WARMUP_ROUNDS}, "
        f"timing starts from round {WARMUP_ROUNDS + 1}",
        flush=True,
    )
    print(f"[overhead] timing log: {timing_log}", flush=True)

    env = RealEnv(dataset=DATASET)
    model = TimedPollingBayesianOptimization(
        env,
        seed=SEED,
        timing_log_path=timing_log,
    )

    model.init_sample()
    print(f"[overhead] init_sample done, global_round={model.global_round}", flush=True)

    for i in range(ITERATIONS):
        model.step()

    model.print_summary()
