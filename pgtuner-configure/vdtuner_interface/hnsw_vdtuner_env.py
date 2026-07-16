# -*- coding: utf-8 -*-
"""
HNSW-only environment matching the vdtuner RealEnv contract (see auto-configure/vdtuner/utils.py).

- ``names``: [M, efConstruction, ef]
- ``get_state(knob_vals_arr)`` with rows in [0,1]^3
- Returns ndarray shape (N, 3): [mean_precision, RPS, p95_time] (same column semantics as RealEnv / ottertune utils)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

_VDIR = Path(__file__).resolve().parent
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))
from benchmark_runner import run_milvus_benchmark  # noqa: E402

_AUTO_CONFIGURE = _REPO / "auto-configure"
if str(_AUTO_CONFIGURE) not in sys.path:
    sys.path.insert(0, str(_AUTO_CONFIGURE))
from configure import configure_index, configure_system, filter_index_rule, filter_system_rule  # noqa: E402

_KNOB_JSON = Path(__file__).resolve().parent / "hnsw_knobs.json"
_WHOLE_PARAM = _AUTO_CONFIGURE / "whole_param.json"


class KnobStandHNSW:
    """Same scaling contract as auto-configure vdtuner KnobStand (subset)."""

    def __init__(self, path: Union[str, Path]) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.knobs_detail = json.load(f)
        self.names = list(self.knobs_detail.keys())

    def scale_back(self, knob_name: str, zero_one_val: float) -> Tuple[int, int]:
        knob = self.knobs_detail[knob_name]
        if knob["type"] == "integer":
            real_val = zero_one_val * (knob["max"] - knob["min"]) + knob["min"]
            iv = int(round(real_val))
            iv = max(knob["min"], min(knob["max"], iv))
            return iv, iv
        raise TypeError(knob["type"])

    def scale_forward(self, knob_name: str, real_val: int) -> float:
        knob = self.knobs_detail[knob_name]
        if knob["type"] == "integer":
            return (real_val - knob["min"]) / (knob["max"] - knob["min"])
        raise TypeError(knob["type"])


def _defaults_full_milvus_conf() -> dict:
    with open(_WHOLE_PARAM, "r", encoding="utf-8") as f:
        meta = json.load(f)
    conf = {k: v["default"] for k, v in meta.items()}
    conf["index_type"] = "HNSW"
    return conf


def apply_hnsw_triplet(m: int, ef_construction: int, ef: int) -> None:
    with open(_WHOLE_PARAM, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    m = max(meta["M"]["min"], min(meta["M"]["max"], int(m)))
    ef_construction = max(
        meta["efConstruction"]["min"],
        min(meta["efConstruction"]["max"], int(ef_construction)),
    )
    ef = max(meta["ef"]["min"], min(meta["ef"]["max"], int(ef)))

    conf = _defaults_full_milvus_conf()
    conf["M"] = m
    conf["efConstruction"] = ef_construction
    conf["ef"] = ef

    idx_copy = dict(conf)
    configure_index(*filter_index_rule(idx_copy))
    system_conf = {k: conf[k] for k in conf if "*" in k}
    configure_system(filter_system_rule(system_conf))


class HNSWVDTunerEnv:
    """
    vdtuner-compatible env for PGTuner HNSW triplets + Milvus benchmark.
    """

    def __init__(
        self,
        benchmark_dataset: str,
        knob_path: Optional[Union[str, Path]] = None,
        engine: str = "milvus-single-node",
        process_tag: str = "milvus-p10",
        use_sudo: bool = True,
        timeout_sec: int = 2000,
    ) -> None:
        self.benchmark_dataset = benchmark_dataset
        self.engine = engine
        self.process_tag = process_tag
        self.use_sudo = use_sudo
        self.timeout_sec = timeout_sec
        path = knob_path or _KNOB_JSON
        self.knob_stand = KnobStandHNSW(path)
        self.names: List[str] = self.knob_stand.names
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0
        self.X_record: list = []
        self.Y1_record: list = []
        self.Y2_record: list = []
        self.Y4_record: list = []

    def get_state(self, knob_vals_arr: np.ndarray) -> np.ndarray:
        """Return columns [precision, RPS, wall_s] compatible with vdtuner RealEnv."""
        y1_list, y4_list, y3_list = [], [], []
        for record in knob_vals_arr:
            conf_vals = [
                self.knob_stand.scale_back(self.names[j], float(record[j]))[1] for j in range(len(self.names))
            ]
            m, efc, ef = conf_vals
            try:
                apply_hnsw_triplet(m, efc, ef)
                prec, rps, p95 = run_milvus_benchmark(
                    self.engine,
                    self.process_tag,
                    self.benchmark_dataset,
                    use_sudo=self.use_sudo,
                    timeout_sec=self.timeout_sec,
                )
            except Exception as e:
                print(f"Benchmark failed: {e}")
                traceback.print_exc()
                prec, rps, p95 = 0.01, 0.01, 1e6

            y1_list.append(prec)
            y4_list.append(rps)
            y3_list.append(int(time.time() - self.t2))
            self.t2 = time.time()
            self.sampled_times += 1
            print(
                f"[{self.sampled_times}] {int(self.t2 - self.t1)} precision={prec} rps={rps} p95={p95}",
                flush=True,
            )

        return np.array([y1_list, y4_list, y3_list]).T

    def default_conf(self) -> List[float]:
        return [self.knob_stand.scale_forward(k, self.knob_stand.knobs_detail[k]["default"]) for k in self.names]
