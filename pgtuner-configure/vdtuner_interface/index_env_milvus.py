# -*- coding: utf-8 -*-
"""
PGTuner IndexEnv with real Milvus measurements via run_engine_test.sh.

Performance vector matches IndexEnv / PCR: [recall, average_construct_dc_counts, average_search_dc_counts].
Recall from benchmark mean precision; search cost uses p95 latency (lower is better, consistent with DC counts).
Construct column is not reported by the benchmark — set to 0 (reward logic uses columns 0 and 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_PCR = Path(__file__).resolve().parent.parent / "parameter_configuration_recommend"
_REPO = Path(__file__).resolve().parents[2]
_AUTO_CONFIGURE = _REPO / "auto-configure"
_WHOLE_PARAM = _AUTO_CONFIGURE / "whole_param.json"
if str(_PCR) not in sys.path:
    sys.path.insert(0, str(_PCR))
if str(_AUTO_CONFIGURE) not in sys.path:
    sys.path.insert(0, str(_AUTO_CONFIGURE))

from configure import configure_index, configure_system, filter_index_rule, filter_system_rule  # noqa: E402
from index_env import IndexEnv  # noqa: E402

_VDIR = Path(__file__).resolve().parent
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))
from benchmark_runner import run_milvus_benchmark  # noqa: E402


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


class IndexEnvMilvus(IndexEnv):
    """
    Same TD3 state / reward as PGTuner IndexEnv, but _get_index_performance runs Milvus benchmark.
    """

    def __init__(
        self,
        *args,
        benchmark_dataset: str,
        engine: str = "milvus-single-node",
        process_tag: str = "milvus-p10",
        use_sudo: bool = True,
        timeout_sec: int = 2000,
        whole_param_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benchmark_dataset = benchmark_dataset
        self.engine = engine
        self.process_tag = process_tag
        self.use_sudo = use_sudo
        self.timeout_sec = timeout_sec
        self.whole_param_path = Path(whole_param_path) if whole_param_path else _WHOLE_PARAM
        with open(self.whole_param_path, "r", encoding="utf-8") as f:
            self._knob_meta: Dict = json.load(f)

    def _defaults_full_conf(self) -> Dict:
        conf = {k: v["default"] for k, v in self._knob_meta.items()}
        conf["index_type"] = "HNSW"
        return conf

    def _apply_hnsw_triplet(self, m: int, ef_construction: int, ef: int) -> None:
        conf = self._defaults_full_conf()

        m = _clip_int(m, self._knob_meta["M"]["min"], self._knob_meta["M"]["max"])
        ef_construction = _clip_int(
            ef_construction,
            self._knob_meta["efConstruction"]["min"],
            self._knob_meta["efConstruction"]["max"],
        )
        ef = _clip_int(ef, self._knob_meta["ef"]["min"], self._knob_meta["ef"]["max"])

        conf["M"] = m
        conf["efConstruction"] = ef_construction
        conf["ef"] = ef

        conf_for_index = dict(conf)
        configure_index(*filter_index_rule(conf_for_index))

        system_conf = {k: conf[k] for k in conf if "*" in k}
        configure_system(filter_system_rule(system_conf))

    def _get_index_performance(self, feature_input):
        """
        feature_input: (N, 9) — first 3 columns are log10(M), efConstruction, log10(efSearch) as in IndexEnv._step.
        """
        n = feature_input.shape[0]
        out = np.zeros((n, 3), dtype=np.float64)
        m_raw = np.power(10.0, feature_input[:, 0])
        ef_c_raw = feature_input[:, 1]
        ef_raw = np.power(10.0, feature_input[:, 2])

        for i in range(n):
            self._apply_hnsw_triplet(m_raw[i], ef_c_raw[i], ef_raw[i])
            mean_precision, _rps, p95 = run_milvus_benchmark(
                self.engine,
                self.process_tag,
                self.benchmark_dataset,
                use_sudo=self.use_sudo,
                timeout_sec=self.timeout_sec,
            )
            out[i, 0] = float(mean_precision)
            out[i, 1] = 0.0
            out[i, 2] = float(p95)
        return out
