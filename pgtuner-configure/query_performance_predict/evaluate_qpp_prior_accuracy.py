#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate accuracy of trained prior-shaped QPP model.

It loads:
- prior space from --prior-config
- feature scaler from scaler_paras/feature_standard_vdtuner_prior.npz
- model checkpoint from --checkpoint OR inferred by vdtuner_prior_qpp_meta.json

Then predicts on a train_data-like CSV and reports MAE / MAPE / Q-error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import Direct_Predict_MLP
from scaler_prior_gpu import Scaler_minmax_prior_gpu
from utils import Scaler_minmax_new_gpu, calculate_errors, np2ts
from utils_vdtuner_prior import build_feature_numpy, read_data_prior

# import from vdtuner_interface
import sys

_QPP = Path(__file__).resolve().parent
_VDIR = _QPP.parent / "vdtuner_interface"
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))
from vdtuner_prior_space import load_prior_space  # noqa: E402


def _default_checkpoint_from_meta(qpp_dir: Path, meta_path: Path) -> Path:
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ls = str(meta["dipredict_layer_sizes"])
    ne = int(meta["dipredict_n_epochs"])
    bs = int(meta["dipredict_batch_size"])
    lr = float(meta["dipredict_lr"])
    ckpt_name = f"{ls}_{ne}_{bs}_{lr}_checkpoint.pth"
    return qpp_dir / "model_checkpoints" / ckpt_name


def _load_model_only(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-config", type=Path, required=True)
    ap.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("../Data/train_data_qpp_prior.csv"),
        help="CSV with required prior QPP columns (FileName + knobs + features + performance)",
    )
    ap.add_argument("--file-name", type=str, default="", help="Optional FileName filter, e.g. vdb_random-100-...")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional QPP checkpoint path")
    ap.add_argument(
        "--meta-json",
        type=Path,
        default=Path("scaler_paras/vdtuner_prior_qpp_meta.json"),
        help="Used to infer checkpoint path when --checkpoint is omitted",
    )
    ap.add_argument(
        "--scaler-npz",
        type=Path,
        default=Path("scaler_paras/feature_standard_vdtuner_prior.npz"),
        help="Feature scaler for prior-shaped QPP",
    )
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output-csv", type=Path, default=Path("qpp_prior_eval_predictions.csv"))
    args = ap.parse_args()

    qpp_dir = _QPP
    prior_config = args.prior_config.resolve()
    eval_csv = args.eval_csv.resolve()
    meta_json = (qpp_dir / args.meta_json).resolve() if not args.meta_json.is_absolute() else args.meta_json
    scaler_npz = (qpp_dir / args.scaler_npz).resolve() if not args.scaler_npz.is_absolute() else args.scaler_npz

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not prior_config.is_file():
        raise FileNotFoundError(f"Missing --prior-config: {prior_config}")
    if not eval_csv.is_file():
        raise FileNotFoundError(f"Missing --eval-csv: {eval_csv}")
    if not scaler_npz.is_file():
        raise FileNotFoundError(f"Missing --scaler-npz: {scaler_npz}")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = _default_checkpoint_from_meta(qpp_dir, meta_json)
    else:
        ckpt_path = ckpt_path.resolve()

    space = load_prior_space(prior_config_path=prior_config)
    df = pd.read_csv(eval_csv)
    if args.file_name:
        df = df[df["FileName"] == args.file_name].copy()
    if len(df) == 0:
        raise RuntimeError("No rows to evaluate after filtering.")

    df_f, df_p = read_data_prior(df, space)
    feat_np = build_feature_numpy(df_f.drop(columns=["FileName"]), space)
    feat_t = np2ts(feat_np).to(device)

    # Ground truth in raw scale.
    gt_raw_np = df_p.to_numpy(dtype=np.float32)
    gt_raw = np2ts(gt_raw_np).to(device)

    head_lo, head_hi = space.wide_min_max_head()
    head_lo_t = torch.from_numpy(head_lo).to(device)
    head_hi_t = torch.from_numpy(head_hi).to(device)
    feature_scaler = Scaler_minmax_prior_gpu(space.wide_dim, head_lo_t, head_hi_t, device)
    feature_scaler.load_parameters(None, str(scaler_npz), device)
    feat_norm = feature_scaler.transform(feat_t)

    model = Direct_Predict_MLP([space.qpp_input_dim, 128, 256, 64, 3])
    # If meta exists, use exact layer sizes from training.
    if meta_json.is_file():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        model = Direct_Predict_MLP(meta["dipredict_layer_sizes"])
    _load_model_only(model, ckpt_path, device)

    perf_scaler = Scaler_minmax_new_gpu(0, device)
    with torch.no_grad():
        pred_n = model(feat_norm)
        pred_raw = perf_scaler.inverse_transform(pred_n)
        pred_raw[:, 1:] = torch.pow(10, pred_raw[:, 1:])

    mae, mape, qerr = calculate_errors(gt_raw, pred_raw)

    print("==== QPP Prior Accuracy ====")
    print(f"rows: {len(df)}")
    print(f"file_filter: {args.file_name or '<all>'}")
    print(f"checkpoint: {ckpt_path}")
    print(f"scaler: {scaler_npz}")
    print(
        "MAE(recall,ct,st): {:.6f}, {:.6f}, {:.6f}".format(
            float(mae[0]), float(mae[1]), float(mae[2])
        )
    )
    print(
        "MAPE(recall,ct,st): {:.6f}, {:.6f}, {:.6f}".format(
            float(mape[0]), float(mape[1]), float(mape[2])
        )
    )
    print(
        "QError(recall,ct,st): {:.6f}, {:.6f}, {:.6f}".format(
            float(qerr[0]), float(qerr[1]), float(qerr[2])
        )
    )

    out = df[["FileName"]].copy()
    gt_np = gt_raw.detach().cpu().numpy()
    pr_np = pred_raw.detach().cpu().numpy()
    out["gt_recall"] = gt_np[:, 0]
    out["gt_average_construct_dc_counts"] = gt_np[:, 1]
    out["gt_average_search_dc_counts"] = gt_np[:, 2]
    out["pred_recall"] = pr_np[:, 0]
    out["pred_average_construct_dc_counts"] = pr_np[:, 1]
    out["pred_average_search_dc_counts"] = pr_np[:, 2]
    out["abs_err_recall"] = np.abs(out["gt_recall"] - out["pred_recall"])
    out["abs_err_average_construct_dc_counts"] = np.abs(
        out["gt_average_construct_dc_counts"] - out["pred_average_construct_dc_counts"]
    )
    out["abs_err_average_search_dc_counts"] = np.abs(
        out["gt_average_search_dc_counts"] - out["pred_average_search_dc_counts"]
    )
    output_csv = args.output_csv if args.output_csv.is_absolute() else (qpp_dir / args.output_csv)
    out.to_csv(output_csv, index=False)
    print(f"Saved per-row predictions to: {output_csv}")


if __name__ == "__main__":
    main()
