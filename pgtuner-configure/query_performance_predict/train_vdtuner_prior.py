# -*- coding: utf-8 -*-
"""
Train QPP (Direct_Predict_MLP) when inputs = VDTuner prior knobs (wide) + dataset features.

Expects ``../Data/train_data.csv`` with columns from ``VDTunerPriorSpace.required_train_columns()``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

_QPP = Path(__file__).resolve().parent
_VDIR = _QPP.parent / "vdtuner_interface"
sys.path.insert(0, str(_QPP))
sys.path.insert(0, str(_VDIR))

from Args import args as a  # noqa: E402
from models import Direct_Predict_MLP  # noqa: E402
from scaler_prior_gpu import Scaler_minmax_prior_gpu  # noqa: E402
from trainer import dipredict_train  # noqa: E402
from utils import CustomDataset, Scaler_minmax_new_gpu, calculate_errors  # noqa: E402
from utils_vdtuner_prior import prepare_train_tensors  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402


def _extra_cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prior-config", type=str, required=True, help="JSON with index_types, tune_knobs, overrides")
    p.add_argument("--knob-json", type=str, default="", help="Override path to whole_param.json")
    p.add_argument(
        "--dipredict-layer-sizes-auto",
        action="store_true",
        help="Set first MLP width to qpp_input_dim (3 outputs unchanged)",
    )
    return p.parse_known_args()[0]


def main() -> None:
    xa = _extra_cli()
    torch.autograd.set_detect_anomaly(True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = str(_QPP)
    parent_directory = str(_QPP.parent)

    os.makedirs(os.path.join(current_directory, "scaler_paras"), exist_ok=True)
    os.makedirs(os.path.join(current_directory, "model_checkpoints"), exist_ok=True)

    knob_path = Path(xa.knob_json) if xa.knob_json else None
    space = load_prior_space(knob_path, prior_config_path=Path(xa.prior_config))

    data_path = os.path.join(parent_directory, "Data/train_data.csv")
    feature_standard_path = os.path.join(current_directory, "scaler_paras/feature_standard_vdtuner_prior.npz")

    print("prior qpp_input_dim:", space.qpp_input_dim, "tuned knobs:", space.knob_names)

    f_train, f_valid, f_test, p_train, p_valid, p_test = prepare_train_tensors(data_path, space, device)

    performance_valid_raw = p_valid.clone()
    performance_test_raw = p_test.clone()
    performance_valid_raw[:, 1:] = torch.pow(10, performance_valid_raw[:, 1:])
    performance_test_eval = performance_test_raw.clone()
    performance_test_eval[:, 1:] = torch.pow(10, performance_test_eval[:, 1:])

    head_lo, head_hi = space.wide_min_max_head()
    head_lo_t = torch.from_numpy(head_lo).to(device)
    head_hi_t = torch.from_numpy(head_hi).to(device)

    feature_scaler = Scaler_minmax_prior_gpu(space.wide_dim, head_lo_t, head_hi_t, device)
    if os.path.exists(feature_standard_path):
        feature_scaler.load_parameters(None, feature_standard_path, device)
    else:
        feature_raw = torch.cat((f_train, f_valid, f_test), dim=0)
        feature_scaler.fit(feature_raw)
        feature_scaler.save_parameters(None, feature_standard_path)

    feature_train = feature_scaler.transform(f_train)
    feature_valid = feature_scaler.transform(f_valid)
    feature_test = feature_scaler.transform(f_test)

    performance_scaler = Scaler_minmax_new_gpu(0, device)
    performance_train = performance_scaler.transform(p_train)
    performance_valid = performance_scaler.transform(p_valid)

    layer_sizes = list(eval(a.dipredict_layer_sizes))
    if xa.dipredict_layer_sizes_auto:
        layer_sizes[0] = space.qpp_input_dim
    if layer_sizes[0] != space.qpp_input_dim:
        print(
            f"Warning: first layer width {layer_sizes[0]} != qpp_input_dim {space.qpp_input_dim}. "
            "Use --dipredict-layer-sizes-auto or fix --dipredict-layer-sizes.",
        )

    dipredict_model_save_path = os.path.join(
        current_directory,
        "model_checkpoints/{}_{}_{}_{}_checkpoint.pth".format(
            str(layer_sizes), a.dipredict_n_epochs, a.dipredict_batch_size, a.dipredict_lr
        ),
    )

    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=a.dipredict_batch_size, shuffle=True)

    model = Direct_Predict_MLP(layer_sizes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=a.dipredict_lr, weight_decay=a.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    meta = {
        "prior_config": str(Path(xa.prior_config).resolve()),
        "knob_names": space.knob_names,
        "wide_dim": space.wide_dim,
        "qpp_input_dim": space.qpp_input_dim,
        "dipredict_layer_sizes": layer_sizes,
        "dipredict_n_epochs": a.dipredict_n_epochs,
        "dipredict_batch_size": a.dipredict_batch_size,
        "dipredict_lr": a.dipredict_lr,
    }
    meta_path = os.path.join(current_directory, "scaler_paras/vdtuner_prior_qpp_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print("Wrote", meta_path)

    dipredict_train(
        dataloader,
        model,
        optimizer,
        scheduler,
        feature_valid,
        performance_valid,
        performance_valid_raw,
        performance_scaler,
        a,
        dipredict_model_save_path,
        None,
        device,
    )

    predicted = model(feature_test)
    predicted = performance_scaler.inverse_transform(predicted)
    predicted[:, 1:] = torch.pow(10, predicted[:, 1:])
    mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_test_eval, predicted)
    print(f"mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}")


if __name__ == "__main__":
    main()
