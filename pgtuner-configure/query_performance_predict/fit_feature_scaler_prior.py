# -*- coding: utf-8 -*-
"""Fit ``feature_standard_vdtuner_prior.npz`` for PCR transfer (dataset B) with prior-shaped QPP inputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn

_QPP = Path(__file__).resolve().parent
_VDIR = _QPP.parent / "vdtuner_interface"
sys.path.insert(0, str(_QPP))
sys.path.insert(0, str(_VDIR))

from scaler_prior_gpu import Scaler_minmax_prior_gpu  # noqa: E402
from utils_vdtuner_prior import build_feature_numpy, read_data_prior  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--prior-config", type=str, required=True)
    p.add_argument("--knob-json", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cudnn.enabled = False
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knob_path = Path(a.knob_json) if a.knob_json else None
    space = load_prior_space(knob_path, prior_config_path=Path(a.prior_config))

    df = pd.read_csv(a.train_csv, sep=",", header=0)
    df_f, _ = read_data_prior(df, space)
    arr = build_feature_numpy(df_f.drop(columns=["FileName"]), space)
    raw = torch.from_numpy(arr).to(device)

    head_lo, head_hi = space.wide_min_max_head()
    scaler = Scaler_minmax_prior_gpu(space.wide_dim, torch.from_numpy(head_lo).to(device), torch.from_numpy(head_hi).to(device), device)
    scaler.fit(raw)
    os.makedirs(os.path.dirname(a.out_npz) or ".", exist_ok=True)
    scaler.save_parameters(None, a.out_npz)
    print(f"Wrote {a.out_npz} (head_dim={space.wide_dim})")


if __name__ == "__main__":
    main()
