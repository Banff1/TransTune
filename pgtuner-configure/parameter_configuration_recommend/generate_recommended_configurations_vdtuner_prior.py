# -*- coding: utf-8 -*-
"""Dump eval pickles to CSV for prior-shaped paras (dynamic knob columns)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PCR = Path(__file__).resolve().parent
_PGT = _PCR.parent
_VDIR = _PGT / "vdtuner_interface"
sys.path.insert(0, str(_PGT))
sys.path.insert(0, str(_VDIR))
sys.path.insert(0, str(_PCR))


def _extra_cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prior-config", type=str, required=True)
    p.add_argument("--knob-json", type=str, default="")
    return p.parse_known_args()[0]


xa = _extra_cli()

from Args import args as args_r  # noqa: E402
from query_performance_predict.Args import args as args_p  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402


def get_pr_results_prior(
    best_performance_file: str,
    best_paras_file: str,
    result_path: str,
    dataset_name: str,
    target_rec_lis: list,
    knob_names: list,
):
    perf_cols = ["recall", "average_ct_dc_counts", "average_st_dc_counts"]
    columns = list(knob_names) + perf_cols

    with open(best_paras_file, "rb") as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, "rb") as f:
        best_performance = pickle.load(f)

    result = np.concatenate((best_paras, best_performance), axis=1)

    df = pd.DataFrame(result, columns=columns)

    df["FileName"] = [dataset_name] * len(target_rec_lis)
    df["target_recall"] = target_rec_lis

    df.to_csv(result_path, mode="w", index=False)

    config_dic = {}
    gcols = ["target_recall"] + list(knob_names)
    config_df = df[["FileName"] + gcols]
    groups = config_df.groupby("FileName")
    for fn, group in groups:
        config_dic[fn] = group[gcols].values.tolist()

    return config_dic


def _decode_knob_rows(df: pd.DataFrame, space, knob_names: list) -> pd.DataFrame:
    """Decode stored raw knob columns (float/int indices) to human-readable knob values."""
    decoded_rows = []
    for _, row in df.iterrows():
        raw = np.array([row[k] for k in knob_names], dtype=np.float32)
        dec = space.raw_numpy_row_to_value_dict(raw)
        decoded_rows.append(dec)
    out = pd.DataFrame(decoded_rows)
    out.columns = [f"decoded__{c}" for c in out.columns]
    return out


if __name__ == "__main__":
    space = load_prior_space(Path(xa.knob_json) if xa.knob_json else None, prior_config_path=Path(xa.prior_config))
    knob_names = space.knob_names

    current_directory = os.getcwd()
    dataset_name = args_r.dataset_name

    store_dir = "./{}_{}_TD3_prior".format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    experiment_mode = args_p.experiment_mode
    store_subdir = os.path.join(store_dir, "{}".format(experiment_mode))
    os.makedirs(store_subdir, exist_ok=True)
    os.makedirs(os.path.join(store_subdir, "recommend_results"), exist_ok=True)

    expr_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        args_r.epoches,
        args_r.max_steps,
        args_r.batch_size,
        args_r.alr,
        args_r.tau,
        args_r.sigma,
        args_r.delay_time,
        args_r.pec_reward,
        args_r.nochange_steps,
    )

    best_performance_file = os.path.join(
        store_subdir,
        "eval_best_performance_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches),
    )
    best_paras_file = os.path.join(
        store_subdir,
        "eval_best_paras_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches),
    )

    result_path = os.path.join(
        store_subdir,
        "recommend_results/eval_{}_{}_{}_{}.csv".format(expr_name, dataset_name, args_r.test_epoches, args_r.nochange_episodes),
    )

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    config_dic = get_pr_results_prior(
        best_performance_file, best_paras_file, result_path, dataset_name, target_rec_lis, knob_names
    )

    df_raw = pd.read_csv(result_path)
    df_dec = _decode_knob_rows(df_raw, space, knob_names)
    readable_path = os.path.join(
        store_subdir,
        "recommend_results/eval_{}_{}_{}_{}_readable.csv".format(
            expr_name, dataset_name, args_r.test_epoches, args_r.nochange_episodes
        ),
    )
    df_out = pd.concat([df_raw, df_dec], axis=1)
    df_out.to_csv(readable_path, index=False)

    key_cols = ["target_recall", "recall", "average_st_dc_counts"] + [f"decoded__{k}" for k in knob_names]
    key_cols = [c for c in key_cols if c in df_out.columns]
    print(f"The recommended configurations for {dataset_name}:")
    print(df_out[key_cols].to_string(index=False))
    print(f"\nRaw CSV: {result_path}")
    print(f"Readable CSV: {readable_path}")
    print(config_dic)
