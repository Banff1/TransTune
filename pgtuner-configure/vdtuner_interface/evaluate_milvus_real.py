# -*- coding: utf-8 -*-
"""
PGTuner PCR online tuning (TD3 + IndexEnv) with real Milvus runs via run_engine_test.sh.

Mirrors parameter_configuration_recommend/evaluate.py but replaces QPP-predicted performance
with benchmark measurements (recall ≈ mean precision, search cost ≈ p95 latency).

Example (from any working directory):

  cd /path/to/pgtuner-configure/vdtuner_interface
  python evaluate_milvus_real.py --benchmark-dataset random-match-int-2048-angular-no-filters \\
      --dataset-name tiny --experiment-mode main

Requires: trained QPP checkpoint + scalers under query_performance_predict/ (same as evaluate.py),
and sudo/docker access as used by vector-db-benchmark run_engine_test.sh.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

_PGTUNER_ROOT = Path(__file__).resolve().parent.parent
_VDTUNER = Path(__file__).resolve().parent
_PCR = _PGTUNER_ROOT / "parameter_configuration_recommend"

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument(
    "--benchmark-dataset",
    type=str,
    required=True,
    help="Dataset name passed to run_engine_test.sh (4th argument), e.g. glove-100-angular",
)
_pre.add_argument("--engine", type=str, default="milvus-single-node")
_pre.add_argument("--process-tag", type=str, default="milvus-p10", help="Benchmark process label, e.g. milvus-p10")
_pre.add_argument("--no-sudo-benchmark", action="store_true", help="Run benchmark without sudo (e.g. docker group)")
_pre.add_argument("--benchmark-timeout", type=int, default=2000)
_bench_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

os.chdir(_PCR)
sys.path.insert(0, str(_PGTUNER_ROOT))
sys.path.insert(0, str(_PCR))
sys.path.insert(0, str(_VDTUNER))

import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.backends import cudnn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Args import args as args_r  # noqa: E402
from index_env_milvus import IndexEnvMilvus  # noqa: E402
from query_performance_predict.Args import args as args_p  # noqa: E402
from query_performance_predict.utils import df2np  # noqa: E402
from TD3 import TD3  # noqa: E402
from utils import Logger  # noqa: E402

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

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
    print("cuda available:", use_cuda)

    parent_directory = str(_PGTUNER_ROOT)

    filename_dic = {
        "deep1": "deep_1_1_96_1",
        "sift1": "sift_1_1_128_1",
        "glove": "glove_1_1.183514_100",
        "paper": "paper_1_2.029997_200",
        "crawl": "crawl_1_1.989995_300",
        "msong": "msong_0_9.92272_420",
        "nytimes": "nytimes_0_2.9_256",
        "tiny": "tiny_1_1_384",
        "gist": "gist_1_1.0_960",
        "deep10": "deep_2_1_96",
        "sift50": "sift_2_5_128_1",
        "sift2": "sift_1_2_128_1",
        "sift3": "sift_1_3_128_1",
        "sift4": "sift_1_4_128_1",
        "sift5": "sift_1_5_128_1",
        "gist_25": "gist_1_1.0_960_25",
        "gist_50": "gist_1_1.0_960_50",
        "gist_75": "gist_1_1.0_960_75",
        "gist_100": "gist_1_1.0_960_100",
    }

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    num_target_rec = len(target_rec_lis)

    dataset_name = args_r.dataset_name
    filename = filename_dic.get(dataset_name, dataset_name)

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
    def_df = df_ini[
        (df_ini["efConstruction"] == 20)
        & (df_ini["M"] == 4)
        & (df_ini["efSearch"] == args_r.baseline_efsearch)
    ]

    feature_df = def_df[
        [
            "SIZE",
            "q_SIZE",
            "DIM",
            "LID",
            "Sum_K_MinDist",
            "Sum_K_MaxDist",
            "Sum_K_StdDist",
            "q_K_MinRatio",
            "q_K_MeanRatio",
            "q_K_MaxRatio",
            "q_K_StdRatio",
        ]
    ]
    def_performance_df = def_df[["recall", "average_construct_dc_counts", "average_search_dc_counts"]]

    data_feature = df2np(feature_df)
    default_performance = df2np(def_performance_df)
    data_feature[:, 0:2] = np.log10(data_feature[:, 0:2])

    final_data_feature = np.repeat(data_feature, num_target_rec, axis=0)
    final_default_performance = np.repeat(default_performance, num_target_rec, axis=0)

    num_dataset = data_feature.shape[0]
    num_data = num_dataset * num_target_rec

    store_dir = "./{}_{}_TD3_milvus_real".format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
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
        "eval_best_performance_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )
    best_paras_file = os.path.join(
        store_subdir,
        "eval_best_paras_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )

    episode_score_file = os.path.join(
        store_subdir,
        "eval_episode_score_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )
    episode_steps_file = os.path.join(
        store_subdir,
        "eval_episode_steps_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )
    episode_closs_file = os.path.join(
        store_subdir,
        "eval_episode_closs_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )
    episode_aloss_file = os.path.join(
        store_subdir,
        "eval_episode_aloss_{}_{}_{}_{}.pkl".format(
            expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
        ),
    )

    logger = Logger(
        name=args_r.method,
        log_file=os.path.join(
            store_subdir,
            "log/eval_{}_{}_{}_{}.log".format(
                expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches
            ),
        ),
    )

    print("---------------- initialize Milvus real-measurement env ----------------")
    env = IndexEnvMilvus(
        num_dataset,
        final_default_performance,
        target_rec_lis,
        args_r,
        args_p,
        predict_model_save_path,
        standard_path,
        device,
        benchmark_dataset=_bench_args.benchmark_dataset,
        engine=_bench_args.engine,
        process_tag=_bench_args.process_tag,
        use_sudo=not _bench_args.no_sudo_benchmark,
        timeout_sec=_bench_args.benchmark_timeout,
    )

    ddpg_opt = {
        "tau": args_r.tau,
        "alr": args_r.alr,
        "clr": args_r.clr,
        "gamma": 0.9,
        "max_steps": args_r.max_steps,
        "batch_size": args_r.batch_size,
        "memory_size": args_r.memory_size,
        "sigma_decay_rate": args_r.sigma_decay_rate,
        "sigma": args_r.sigma,
        "delay_time": args_r.delay_time,
        "actor_layer_sizes": eval(args_r.actor_layer_sizes),
        "critic_layer_sizes": eval(args_r.critic_layer_sizes),
        "actor_path": os.path.join(store_dir, "model_params/actor_{}.pth".format(expr_name)),
        "critic1_path": os.path.join(store_dir, "model_params/critic1_{}.pth".format(expr_name)),
        "critic2_path": os.path.join(store_dir, "model_params/critic2_{}.pth".format(expr_name)),
    }

    model = TD3(
        n_states=args_r.n_states,
        n_actions=args_r.n_actions,
        num_data=num_data,
        opt=ddpg_opt,
        dv=device,
    )

    episode_score = {}
    episode_steps = {}
    episode_closs = {}
    episode_aloss = {}

    env.nochange_episodes = 0
    start_time = time.time()
    print("---------------- start recommending (each step runs real benchmark × targets) ----------------")
    for episode in tqdm(range(args_r.test_epoches), total=args_r.test_epoches):
        current_states = env._initialize()
        model.reset(args_r.sigma)

        train_step = 0
        accumulate_loss = [0, 0]

        for _st in tqdm(range(args_r.max_steps), total=args_r.max_steps):
            states = current_states
            actions = model.choose_action(states, True)
            rewards, states_, _dones, _, _, _, _ = env._step(
                actions, final_data_feature, best_performance_file, best_paras_file
            )
            model.add_sample(states, actions, rewards, states_, _dones)
            current_states = states_

            if len(model.replay_memory) > args_r.batch_size:
                losses = []
                for _i in range(2):
                    loss = model.update()
                    if (model.update_time % model.delay_time) == 0:
                        losses.append(loss)
                        train_step += 1
                accumulate_loss[0] += sum(x[0] for x in losses)
                accumulate_loss[1] += sum(x[1] for x in losses)

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
        episode_closs[episode] = accumulate_loss[0] / max(1, train_step)
        episode_aloss[episode] = accumulate_loss[1] / max(1, train_step)

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
