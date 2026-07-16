"""Timed wrapper around PollingBayesianOptimization for overhead analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import subprocess as sp

from optimizer_pobo_sa import PollingBayesianOptimization


WARMUP_ROUNDS = 7


class TimedPollingBayesianOptimization(PollingBayesianOptimization):
    """Record per-iteration BO propose time and benchmark time after warmup."""

    def __init__(self, *args, timing_log_path: str | Path | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_round = 0
        self.timing_records: list[dict[str, Any]] = []
        default_path = Path(__file__).resolve().parent / "overhead_timing.jsonl"
        self.timing_log_path = Path(timing_log_path) if timing_log_path else default_path

    def init_sample(self) -> None:
        super().init_sample()
        self.global_round = len(self.polling_index)

    def step(self) -> None:
        self.global_round += 1
        round_num = self.global_round
        track = round_num > WARMUP_ROUNDS

        t_step_start = time.perf_counter()

        t_bo_start = time.perf_counter()
        if len(self.remain_types) > 1:
            self.successive_abandon()
        sp.run(
            f"echo {list(self.chosen_ref_whole)} {list(self.chosen_ref_k.values())}  "
            f"{list(self.delta_hv.values())} {self.worst_type_record[-1]} {self.remain_types}>> pobo_record.log",
            shell=True,
            stdout=sp.PIPE,
        )
        polling_k, new_x = self.rr_polling()
        t_bo_end = time.perf_counter()

        t_bench_start = time.perf_counter()
        new_y = self.env.get_state(new_x)
        t_bench_end = time.perf_counter()

        self.X[polling_k] = self.X[polling_k] + new_x.tolist()
        self.Y[polling_k] = self.Y[polling_k] + new_y.tolist()

        t_update_start = time.perf_counter()
        self.update_model()
        t_update_end = time.perf_counter()

        if not track:
            return

        record = {
            "round": round_num,
            "index_type": polling_k,
            "bo_propose_s": round(t_bo_end - t_bo_start, 3),
            "benchmark_s": round(t_bench_end - t_bench_start, 3),
            "bo_update_s": round(t_update_end - t_update_start, 3),
            "total_s": round(t_update_end - t_step_start, 3),
        }
        self.timing_records.append(record)
        self._append_timing_record(record)
        self._print_timing_record(record)

    def _append_timing_record(self, record: dict[str, Any]) -> None:
        self.timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.timing_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _print_timing_record(record: dict[str, Any]) -> None:
        print(
            f"[overhead] round={record['round']} index_type={record['index_type']} "
            f"bo_propose={record['bo_propose_s']}s "
            f"benchmark={record['benchmark_s']}s "
            f"bo_update={record['bo_update_s']}s "
            f"total={record['total_s']}s",
            flush=True,
        )

    def print_summary(self) -> None:
        if not self.timing_records:
            print(
                f"[overhead] No timed rounds recorded (need round > {WARMUP_ROUNDS}).",
                flush=True,
            )
            return

        n = len(self.timing_records)
        bo_sum = sum(r["bo_propose_s"] for r in self.timing_records)
        bench_sum = sum(r["benchmark_s"] for r in self.timing_records)
        update_sum = sum(r["bo_update_s"] for r in self.timing_records)
        total_sum = sum(r["total_s"] for r in self.timing_records)

        print("[overhead] ===== timing summary (rounds > 7) =====", flush=True)
        print(f"[overhead] timed_rounds={n}", flush=True)
        print(
            f"[overhead] bo_propose: total={bo_sum:.1f}s avg={bo_sum / n:.3f}s",
            flush=True,
        )
        print(
            f"[overhead] benchmark:  total={bench_sum:.1f}s avg={bench_sum / n:.1f}s",
            flush=True,
        )
        print(
            f"[overhead] bo_update:  total={update_sum:.1f}s avg={update_sum / n:.3f}s",
            flush=True,
        )
        print(
            f"[overhead] total:      total={total_sum:.1f}s avg={total_sum / n:.1f}s",
            flush=True,
        )
        print(f"[overhead] log file: {self.timing_log_path}", flush=True)
