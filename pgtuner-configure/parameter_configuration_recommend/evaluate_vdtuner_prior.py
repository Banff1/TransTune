# -*- coding: utf-8 -*-
"""Online evaluation with ``IndexEnvVDTunerPrior`` (matches ``train_vdtuner_prior.py`` TD3_prior layout)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from tqdm import tqdm

_PCR = Path(__file__).resolve().parent
_PGT = _PCR.parent
_VDIR = _PGT / "vdtuner_interface"
sys.path.insert(0, str(_PGT))
sys.path.insert(0, str(_VDIR))
sys.path.insert(0, str(_PCR))


def _baseline_mask(df: pd.DataFrame, baseline: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in baseline.items():
        if k not in df.columns:
            raise ValueError(f"baseline key {k} not in test CSV")
        if isinstance(v, bool):
            m &= df[k].astype(bool) == v
        elif isinstance(v, float):
            m &= np.isclose(df[k].astype(float), float(v))
        else:
            m &= df[k] == v
    return m


def _extra_cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prior-config", type=str, required=True)
    p.add_argument("--knob-json", type=str, default="")
    return p.parse_known_args()[0]


xa = _extra_cli()

from query_performance_predict.Args import args as args_p  # noqa: E402
from query_performance_predict.utils import df2np  # noqa: E402
from TD3 import TD3  # noqa: E402
from Args import args as args_r  # noqa: E402
from index_env_vdtuner_prior import IndexEnvVDTunerPrior, rl_state_action_dims  # noqa: E402
from utils import Logger  # noqa: E402
from vdtuner_prior_space import DATASET_FEATURE_COLUMNS, load_prior_bundle  # noqa: E402

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    import random

    random.seed(args_r.seed)
    np.random.seed(args_r.seed)
    torch.manual_seed(args_r.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_r.seed)
        torch.cuda.manual_seed_all(args_r.seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    knob_path = Path(xa.knob_json) if xa.knob_json else None
    bundle = load_prior_bundle(knob_path, prior_config_path=Path(xa.prior_config))
    space = bundle.space
    baseline = bundle.baseline
    k = space.compact_dim
    n_states, n_actions = rl_state_action_dims(space)
    args_r.n_states = n_states
    args_r.n_actions = n_actions

    baseline_row = np.zeros((len(space.knob_names),), dtype=np.float32)
    for j, name in enumerate(space.knob_names):
        meta = space.knobs_meta[name]
        if meta["type"] == "integer":
            baseline_row[j] = float(int(baseline[name]))
        elif meta["type"] == "enum":
            vals = list(meta["enum_values"])
            if all(type(x) is bool for x in vals):
                baseline_row[j] = 1.0 if bool(baseline[name]) else 0.0
            else:
                baseline_row[j] = float(vals.index(baseline[name]))

    dataset_name = args_r.dataset_name
    filename = dataset_name

    max_selected_num = args_p.max_selected_num
    selected_num = 2
    selected_rounds = max_selected_num // selected_num

    mode = args_p.experiment_mode

    predict_model_save_path = os.path.join(
        parent_directory,
        "query_performance_predict/model_checkpoints/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth".format(
            mode,
            dataset_name,
            args_p.dipredict_layer_sizes,
            args_p.dipredict_n_epochs,
            args_p.dipredict_batch_size,
            args_p.dipredict_lr,
            selected_num,
            selected_rounds,
        ),
    )
    standard_path = os.path.join(
        parent_directory,
        "query_performance_predict/scaler_paras/{}/{}_feature_standard_{}_{}.npz".format(
            mode, dataset_name, selected_num, selected_rounds
        ),
    )

    if mode == "main" or mode == "dataset_change":
        data_path = os.path.join(parent_directory, "Data/test_data_main.csv")
    elif mode == "ds_change":
        data_path = os.path.join(parent_directory, "Data/test_data_ds_change.csv")
    elif mode == "qd_change":
        data_path = os.path.join(parent_directory, "Data/test_data_qd_change.csv")
    else:
        data_path = os.path.join(parent_directory, "Data/test_data_main.csv")

    df = pd.read_csv(data_path, sep=",", header=0)

    df_ini = df[df["FileName"] == filename]
    def_df = df_ini[_baseline_mask(df_ini, baseline)]

    feature_df = def_df[DATASET_FEATURE_COLUMNS]
    def_performance_df = def_df[["recall", "average_construct_dc_counts", "average_search_dc_counts"]]

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    num_target_rec = len(target_rec_lis)

    data_feature = df2np(feature_df)
    default_performance = df2np(def_performance_df)

    data_feature[:, 0:2] = np.log10(data_feature[:, 0:2])

    final_data_feature = np.repeat(data_feature, num_target_rec, axis=0)
    final_default_performance = np.repeat(default_performance, num_target_rec, axis=0)

    num_dataset = data_feature.shape[0]
    num_data = final_data_feature.shape[0]

    store_dir = "./{}_{}_TD3_prior".format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    os.makedirs(store_dir, exist_ok=True)

    store_subdir = os.path.join(store_dir, "{}".format(mode))
    os.makedirs(store_subdir, exist_ok=True)

    for name in ("log", "runs", "save_memory", "model_params"):
        os.makedirs(os.path.join(store_subdir, name), exist_ok=True)

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

    episode_score_file = os.path.join(
        store_subdir, "eval_episode_score_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)
    )
    episode_steps_file = os.path.join(
        store_subdir, "eval_episode_steps_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)
    )
    episode_closs_file = os.path.join(
        store_subdir, "eval_episode_closs_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)
    )
    episode_aloss_file = os.path.join(
        store_subdir, "eval_episode_aloss_{}_{}_{}_{}.pkl".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)
    )

    logger = Logger(
        name=args_r.method,
        log_file=os.path.join(store_subdir, "log/eval_{}_{}_{}_{}.log".format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)),
    )

    print("----------------initialize IndexEnvVDTunerPrior----------------")
    env = IndexEnvVDTunerPrior(
        space,
        num_dataset,
        final_default_performance,
        target_rec_lis,
        args_r,
        args_p,
        predict_model_save_path,
        standard_path,
        baseline_row,
        device,
    )

    print("----------------load TD3 (prior)----------------")
    ddpg_opt = dict()
    ddpg_opt["tau"] = args_r.tau
    ddpg_opt["alr"] = args_r.alr
    ddpg_opt["clr"] = args_r.clr
    ddpg_opt["gamma"] = 0.9
    ddpg_opt["max_steps"] = args_r.max_steps
    ddpg_opt["batch_size"] = args_r.batch_size
    ddpg_opt["memory_size"] = args_r.memory_size
    ddpg_opt["sigma_decay_rate"] = args_r.sigma_decay_rate
    ddpg_opt["sigma"] = args_r.sigma
    ddpg_opt["delay_time"] = args_r.delay_time
    ddpg_opt["actor_layer_sizes"] = eval(args_r.actor_layer_sizes)
    ddpg_opt["critic_layer_sizes"] = eval(args_r.critic_layer_sizes)
    ddpg_opt["actor_path"] = os.path.join(store_dir, "model_params/actor_{}.pth".format(expr_name))
    ddpg_opt["critic1_path"] = os.path.join(store_dir, "model_params/critic1_{}.pth".format(expr_name))
    ddpg_opt["critic2_path"] = os.path.join(store_dir, "model_params/critic2_{}.pth".format(expr_name))

    model = TD3(n_states=n_states, n_actions=n_actions, num_data=num_data, opt=ddpg_opt, dv=device)

    episode_score = {}
    episode_steps = {}
    episode_closs = {}
    episode_aloss = {}

    start_time = time.time()
    print("----------------start recommending (prior)...----------------")
    for episode in tqdm(range(args_r.test_epoches), total=args_r.test_epoches):
        current_states = env._initialize()

        model.reset(args_r.sigma)

        train_step = 0
        accumulate_loss = [0, 0]

        for st in tqdm(range(args_r.max_steps), total=args_r.max_steps):
            states = current_states
            actions = model.choose_action(states, True)

            rewards, states_, dones, _, _, _, _ = env._step(actions, final_data_feature, best_performance_file, best_paras_file)

            next_states = states_

            model.add_sample(states, actions, rewards, next_states, dones)

            current_states = next_states

            if len(model.replay_memory) > args_r.batch_size:
                losses = []
                for i in range(2):
                    loss = model.update()
                    if (model.update_time % model.delay_time) == 0:
                        losses.append(loss)
                        train_step += 1

                accumulate_loss[0] += sum([x[0] for x in losses])
                accumulate_loss[1] += sum([x[1] for x in losses])

            if env.nochange_steps == args_r.nochange_steps:
                break

        if env.steps == args_r.nochange_steps:
            env.nochange_episodes += 1
        else:
            env.nochange_episodes = 0

        if env.nochange_episodes == args_r.nochange_episodes:
            break

        model.actor_scheduler.step()
        model.critic1_scheduler.step()
        model.critic2_scheduler.step()

        episode_score[episode] = env.score
        episode_steps[episode] = env.steps
        episode_closs[episode] = accumulate_loss[0] / max(train_step, 1)
        episode_aloss[episode] = accumulate_loss[1] / max(train_step, 1)

        print(f"episode score: {env.score}, steps: {env.steps}")

    with open(episode_score_file, "wb") as f:
        pickle.dump(episode_score, f)
    with open(episode_steps_file, "wb") as f:
        pickle.dump(episode_steps, f)
    with open(episode_closs_file, "wb") as f:
        pickle.dump(episode_closs, f)
    with open(episode_aloss_file, "wb") as f:
        pickle.dump(episode_aloss, f)

    logger.info("recommend time: {}s".format(time.time() - start_time))
