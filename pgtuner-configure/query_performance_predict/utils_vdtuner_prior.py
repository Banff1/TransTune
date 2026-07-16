# -*- coding: utf-8 -*-
"""Load QPP training matrices when each row includes VDTuner prior knobs (index + system)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

_VDIR = Path(__file__).resolve().parent.parent / "vdtuner_interface"
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))

from vdtuner_prior_space import (  # noqa: E402
    DATASET_FEATURE_COLUMNS,
    PERFORMANCE_COLUMNS,
    VDTunerPriorSpace,
)

from utils import df2np, get_dataset, np2ts  # noqa: E402


def read_data_prior(df: pd.DataFrame, space: VDTunerPriorSpace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in space.required_train_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"train CSV missing columns: {missing[:12]}{'...' if len(missing) > 12 else ''}")
    df_f = df[["FileName"] + space.knob_names + DATASET_FEATURE_COLUMNS]
    df_p = df[PERFORMANCE_COLUMNS]
    return df_f, df_p


def build_feature_numpy(df_f: pd.DataFrame, space: VDTunerPriorSpace) -> np.ndarray:
    """Wide prior encoding + dataset columns; applies log10 to SIZE, q_SIZE (PCR convention)."""
    rows = []
    tail = df_f[DATASET_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    tail[:, 0:2] = np.log10(np.clip(tail[:, 0:2], 1e-12, None))
    for i in range(len(df_f)):
        vals = {name: df_f.iloc[i][name] for name in space.knob_names}
        wide = space.encode_wide_row(vals)
        rows.append(np.concatenate([wide, tail[i]], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def prepare_train_tensors(
    data_path: str,
    space: VDTunerPriorSpace,
    device: torch.device,
) -> tuple:
    df_train, df_valid, df_test = get_dataset(data_path)
    df_train_f, df_train_p = read_data_prior(df_train, space)
    df_valid_f, df_valid_p = read_data_prior(df_valid, space)
    df_test_f, df_test_p = read_data_prior(df_test, space)

    f_train = build_feature_numpy(df_train_f.drop(columns=["FileName"]), space)
    f_valid = build_feature_numpy(df_valid_f.drop(columns=["FileName"]), space)
    f_test = build_feature_numpy(df_test_f.drop(columns=["FileName"]), space)

    p_train = df2np(df_train_p)
    p_valid = df2np(df_valid_p)
    p_test = df2np(df_test_p)

    # Same targets as legacy QPP
    p_train[:, 1:] = np.log10(np.clip(p_train[:, 1:], 1e-12, None))
    p_valid[:, 1:] = np.log10(np.clip(p_valid[:, 1:], 1e-12, None))
    p_test[:, 1:] = np.log10(np.clip(p_test[:, 1:], 1e-12, None))

    f_train_t = np2ts(f_train).to(device)
    f_valid_t = np2ts(f_valid).to(device)
    f_test_t = np2ts(f_test).to(device)
    p_train_t = np2ts(p_train).to(device)
    p_valid_t = np2ts(p_valid).to(device)
    p_test_t = np2ts(p_test).to(device)

    return (f_train_t, f_valid_t, f_test_t, p_train_t, p_valid_t, p_test_t)
