# -*- coding: utf-8 -*-
"""
Bridge to PGTuner HNSW + Milvus benchmark env (same ``get_state`` / ``default_conf`` as ``RealEnv``).

Usage::

    from hnsw_pgtuner_bridge import HNSWVDTunerEnv
    env = HNSWVDTunerEnv(benchmark_dataset=\"glove-100-angular\")
    env.get_state(np.array([[0.5, 0.5, 0.5]]))  # -> [[precision, rps, time], ...]

Full QPP → PCR → recommend: run ``pgtuner-configure/vdtuner_interface/run_pgtuner_milvus_pipeline.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PGT = Path(__file__).resolve().parents[2] / "pgtuner-configure" / "vdtuner_interface"
if str(_PGT) not in sys.path:
    sys.path.insert(0, str(_PGT))

from hnsw_vdtuner_env import HNSWVDTunerEnv  # noqa: E402

__all__ = ["HNSWVDTunerEnv"]
