# -*- coding: utf-8 -*-
"""
Fit and save feature_scaler (Scaler_minmax_new_gpu) from train_data.csv only.
Used when PCR runs on dataset B while QPP was trained on A — tail mean/std must match B.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.backends import cudnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import df2np, np2ts, Scaler_minmax_new_gpu
from utils import read_data_new


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--out-npz", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cudnn.enabled = False
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd

    df = pd.read_csv(a.train_csv, sep=",", header=0)
    df_f, _df_p = read_data_new(df)
    df_f = df_f.drop(["FileName"], axis=1)

    feature_raw = df2np(df_f)
    feature_raw[:, 0] = np.log10(feature_raw[:, 0])
    feature_raw[:, 2:5] = np.log10(feature_raw[:, 2:5])
    feature_raw = np2ts(feature_raw).to(device)

    feature_scaler = Scaler_minmax_new_gpu(6, device)
    feature_scaler.fit(feature_raw)
    os.makedirs(os.path.dirname(a.out_npz) or ".", exist_ok=True)
    feature_scaler.save_parameters(None, a.out_npz)
    print(f"Wrote {a.out_npz}")


if __name__ == "__main__":
    main()
