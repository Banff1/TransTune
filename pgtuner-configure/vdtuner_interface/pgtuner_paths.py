# -*- coding: utf-8 -*-
"""Align QPP training outputs with PCR evaluate.py / IndexEnv expected paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def pgtuner_root() -> Path:
    return Path(__file__).resolve().parent.parent


def qpp_checkpoint_after_train(
    dipredict_layer_sizes: str,
    dipredict_n_epochs: int,
    dipredict_batch_size: int,
    dipredict_lr: float,
) -> Path:
    """Flat path written by query_performance_predict/train.py."""
    qpp = pgtuner_root() / "query_performance_predict"
    name = "{}_{}_{}_{}_checkpoint.pth".format(
        dipredict_layer_sizes,
        dipredict_n_epochs,
        dipredict_batch_size,
        dipredict_lr,
    )
    return qpp / "model_checkpoints" / name


def qpp_feature_scaler_after_train() -> Path:
    return pgtuner_root() / "query_performance_predict" / "scaler_paras" / "feature_standard.npz"


def qpp_checkpoint_for_evaluate(
    mode: str,
    dataset_name: str,
    dipredict_layer_sizes: str,
    dipredict_n_epochs: int,
    dipredict_batch_size: int,
    dipredict_lr: float,
    max_selected_num: int,
) -> Path:
    selected_num = 2
    selected_rounds = max_selected_num // selected_num
    qpp = pgtuner_root() / "query_performance_predict"
    name = "{}_{}_{}_{}_{}_{}_{}_checkpoint.pth".format(
        dataset_name,
        dipredict_layer_sizes,
        dipredict_n_epochs,
        dipredict_batch_size,
        dipredict_lr,
        selected_num,
        selected_rounds,
    )
    return qpp / "model_checkpoints" / mode / name


def qpp_scaler_for_evaluate(
    mode: str,
    dataset_name: str,
    max_selected_num: int,
) -> Path:
    selected_num = 2
    selected_rounds = max_selected_num // selected_num
    qpp = pgtuner_root() / "query_performance_predict"
    name = "{}_feature_standard_{}_{}.npz".format(dataset_name, selected_num, selected_rounds)
    return qpp / "scaler_paras" / mode / name


def sync_qpp_artifacts_for_pcr_evaluate(
    mode: str,
    dataset_name: str,
    *,
    dipredict_layer_sizes: str,
    dipredict_n_epochs: int,
    dipredict_batch_size: int,
    dipredict_lr: float,
    max_selected_num: int,
) -> None:
    """Copy train.py outputs to the locations load_model() uses in evaluate.py / IndexEnv."""
    src_ckpt = qpp_checkpoint_after_train(
        dipredict_layer_sizes, dipredict_n_epochs, dipredict_batch_size, dipredict_lr
    )
    dst_ckpt = qpp_checkpoint_for_evaluate(
        mode,
        dataset_name,
        dipredict_layer_sizes,
        dipredict_n_epochs,
        dipredict_batch_size,
        dipredict_lr,
        max_selected_num,
    )
    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Train QPP first; missing checkpoint: {src_ckpt}")
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)

    src_scaler = qpp_feature_scaler_after_train()
    dst_scaler = qpp_scaler_for_evaluate(mode, dataset_name, max_selected_num)
    if not src_scaler.is_file():
        raise FileNotFoundError(f"Missing feature scaler (run train.py): {src_scaler}")
    dst_scaler.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_scaler, dst_scaler)


def sync_qpp_transfer_to_pcr_evaluate(
    mode: str,
    pcr_dataset_name: str,
    scaler_pcr_npz: Path,
    *,
    dipredict_layer_sizes: str,
    dipredict_n_epochs: int,
    dipredict_batch_size: int,
    dipredict_lr: float,
    max_selected_num: int,
) -> None:
    """
    PCR evaluate loads QPP weights trained on dataset A but must use feature scaler fit on dataset B.
    Copy flat QPP checkpoint -> main/{pcr}_*.pth; copy scaler_pcr_npz -> main/{pcr}_feature_standard_*.npz.
    """
    src_ckpt = qpp_checkpoint_after_train(
        dipredict_layer_sizes, dipredict_n_epochs, dipredict_batch_size, dipredict_lr
    )
    dst_ckpt = qpp_checkpoint_for_evaluate(
        mode,
        pcr_dataset_name,
        dipredict_layer_sizes,
        dipredict_n_epochs,
        dipredict_batch_size,
        dipredict_lr,
        max_selected_num,
    )
    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Missing QPP checkpoint: {src_ckpt}")
    scaler_pcr_npz = Path(scaler_pcr_npz)
    if not scaler_pcr_npz.is_file():
        raise FileNotFoundError(f"Missing PCR feature scaler: {scaler_pcr_npz}")
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)
    dst_scaler = qpp_scaler_for_evaluate(mode, pcr_dataset_name, max_selected_num)
    dst_scaler.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scaler_pcr_npz, dst_scaler)


def copy_flat_scaler_for_pcr_train(scaler_pcr_npz: Path) -> None:
    """PCR train.py loads scaler_paras/feature_standard.npz (flat path)."""
    dst = pgtuner_root() / "query_performance_predict" / "scaler_paras" / "feature_standard.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scaler_pcr_npz, dst)


def vdtuner_prior_qpp_meta_path() -> Path:
    return pgtuner_root() / "query_performance_predict" / "scaler_paras" / "vdtuner_prior_qpp_meta.json"


def load_vdtuner_prior_qpp_meta() -> dict:
    p = vdtuner_prior_qpp_meta_path()
    if not p.is_file():
        raise FileNotFoundError(f"Missing prior QPP meta (train_vdtuner_prior.py): {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def qpp_checkpoint_after_train_prior() -> Path:
    """Checkpoint path written by ``train_vdtuner_prior.py`` (uses str(layer_sizes))."""
    meta = load_vdtuner_prior_qpp_meta()
    ls = meta["dipredict_layer_sizes"]
    qpp = pgtuner_root() / "query_performance_predict"
    name = "{}_{}_{}_{}_checkpoint.pth".format(
        str(ls),
        meta["dipredict_n_epochs"],
        meta["dipredict_batch_size"],
        meta["dipredict_lr"],
    )
    return qpp / "model_checkpoints" / name


def sync_qpp_transfer_to_pcr_evaluate_prior(
    mode: str,
    pcr_dataset_name: str,
    scaler_pcr_npz: Path,
    *,
    max_selected_num: int,
) -> None:
    """Like ``sync_qpp_transfer_to_pcr_evaluate`` but uses prior QPP checkpoint + meta."""
    meta = load_vdtuner_prior_qpp_meta()
    ls = str(meta["dipredict_layer_sizes"])
    ne = int(meta["dipredict_n_epochs"])
    bs = int(meta["dipredict_batch_size"])
    lr = float(meta["dipredict_lr"])
    src_ckpt = qpp_checkpoint_after_train_prior()
    dst_ckpt = qpp_checkpoint_for_evaluate(mode, pcr_dataset_name, ls, ne, bs, lr, max_selected_num)
    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Missing prior QPP checkpoint: {src_ckpt}")
    scaler_pcr_npz = Path(scaler_pcr_npz)
    if not scaler_pcr_npz.is_file():
        raise FileNotFoundError(f"Missing PCR feature scaler: {scaler_pcr_npz}")
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)
    dst_scaler = qpp_scaler_for_evaluate(mode, pcr_dataset_name, max_selected_num)
    dst_scaler.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scaler_pcr_npz, dst_scaler)
