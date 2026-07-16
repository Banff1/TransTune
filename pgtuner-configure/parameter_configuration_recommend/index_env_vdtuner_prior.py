# -*- coding: utf-8 -*-
"""
PCR environment when QPP was trained with ``train_vdtuner_prior.py`` (wide knobs + dataset tail).

State dim = 2 * K + 6 (current/best compact [0,1]^K, recall + log10(search DC), four delta terms).
Action dim = K (one sigmoid per tuned knob, same semantics as VDTuner ``KnobStand``).
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.optim as optim

_VDIR = Path(__file__).resolve().parent.parent / "vdtuner_interface"
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))

from query_performance_predict.models import Direct_Predict_MLP  # noqa: E402
from query_performance_predict.scaler_prior_gpu import Scaler_minmax_prior_gpu  # noqa: E402
from query_performance_predict.utils import load_model, np2ts  # noqa: E402
from query_performance_predict.utils import Scaler_minmax_new_gpu  # noqa: E402
from vdtuner_prior_space import VDTunerPriorSpace  # noqa: E402


class ScalerStateVDTunerPrior:
    """Min–max state normalization mirroring ``Scaler_state`` layout for variable K."""

    def __init__(self, k: int):
        self.k = k
        self.dim = 2 * k + 6
        self.min = np.zeros(self.dim, dtype=np.float64)
        self.max = np.ones(self.dim, dtype=np.float64)
        # recall, log10(search DC counts) — same tail as legacy Scaler_state indices 6–7
        self.min[2 * k + 1] = 2.0
        self.max[2 * k + 1] = np.log10(500000)
        self.min[2 * k + 2 : 2 * k + 6] = np.array([0.0, 0.0, 0.0, -1000.0])
        self.max[2 * k + 2 : 2 * k + 6] = np.array([1.0, 1.0, 1.0, 1.0])

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min + 1e-12)


def rl_state_action_dims(space: VDTunerPriorSpace) -> Tuple[int, int]:
    k = space.compact_dim
    return 2 * k + 6, k


class IndexEnvVDTunerPrior:
    def __init__(
        self,
        space: VDTunerPriorSpace,
        num_dataset: int,
        default_performance: np.ndarray,
        target_rec_lis: List[float],
        args_r: Any,
        args_p: Any,
        predict_model_save_path: str,
        standard_path: str,
        baseline_raw_row: np.ndarray,
        dv: torch.device,
    ):
        self.device = dv
        self.space = space
        self.K = space.compact_dim
        self.args_p = args_p
        self.args_r = args_r

        layer_sizes = eval(args_p.dipredict_layer_sizes)
        self.predict_model = Direct_Predict_MLP(layer_sizes).to(self.device)
        self._get_predict_model(predict_model_save_path)

        head_lo, head_hi = space.wide_min_max_head()
        self.feature_scaler = Scaler_minmax_prior_gpu(
            space.wide_dim,
            torch.from_numpy(head_lo).to(dv),
            torch.from_numpy(head_hi).to(dv),
            dv,
        )
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.state_scaler = ScalerStateVDTunerPrior(self.K)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))

        self.score = 0.0
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0
        self.nochange_episodes = 0

        br = baseline_raw_row.reshape(1, -1).astype(np.float32)
        if br.shape[1] != self.K:
            raise ValueError(f"baseline_raw_row width {br.shape[1]} != K={self.K}")
        self.default_paras = np.tile(br, (self.target_rec.shape[0], 1))
        self.default_index_performance = default_performance.copy()
        self.best_index_performance = default_performance.copy()

    def _get_predict_model(self, model_save_path: str) -> None:
        optimizer = optim.Adam(
            self.predict_model.parameters(), lr=self.args_p.dipredict_lr, weight_decay=self.args_p.weight_decay
        )
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path: str) -> None:
        self.feature_scaler.load_parameters(None, standard_path, self.device)

    def _raw_to_compact(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw, dtype=np.float64)
        for i in range(raw.shape[0]):
            d = self.space.raw_numpy_row_to_value_dict(raw[i])
            for j, name in enumerate(self.space.knob_names):
                out[i, j] = self.space.scale_forward_compact(name, d[name])
        return out

    def _get_action(self, actions: np.ndarray) -> np.ndarray:
        return self.space.actions_to_raw(actions)

    def _wide_feature_block(self, cur_paras: np.ndarray, data_feature: np.ndarray) -> np.ndarray:
        num = cur_paras.shape[0]
        blocks = []
        for i in range(num):
            d = self.space.raw_numpy_row_to_value_dict(cur_paras[i])
            w = self.space.encode_wide_row(d)
            blocks.append(np.concatenate([w, data_feature[i]], axis=0))
        return np.stack(blocks, axis=0)

    def _get_index_performance(self, feature_input: np.ndarray) -> np.ndarray:
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()
        real_index_performance[:, 1:] = np.clip(real_index_performance[:, 1:], 1e-12, None)
        real_index_performance = np.nan_to_num(real_index_performance, nan=0.0, posinf=1e12, neginf=0.0)
        return real_index_performance

    @staticmethod
    def _get_performance_improvement(last_index_performance: np.ndarray, current_index_performance: np.ndarray) -> np.ndarray:
        denom = np.where(np.abs(last_index_performance[:, 2]) < 1e-12, 1e-12, last_index_performance[:, 2])
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / denom
        st_counts_dec = np.nan_to_num(st_counts_dec, nan=0.0, posinf=0.0, neginf=0.0)
        return st_counts_dec

    def _get_best_now(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            return pickle.load(f)

    def _get_best_paras_now(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            return pickle.load(f)

    def _record_best(self, cur_index_performance: np.ndarray, cur_paras: np.ndarray, performance_filename: str, paras_filename: str) -> np.ndarray:
        if os.path.exists(performance_filename):
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = best_rec < target_rec
            cond2 = best_rec >= target_rec
            cond3 = cur_rec < target_rec
            cond4 = cur_rec >= target_rec
            cond5 = cur_rec > best_rec
            cond6 = target_dec > 0

            cond_a = cond1 & cond4
            cond_b = cond1 & cond3 & cond5
            cond_c = cond2 & cond4 & cond6

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, "wb") as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, "wb") as f:
                    pickle.dump(best_paras, f)
            else:
                self.nochange_steps += 1

            return cond_c
        with open(performance_filename, "wb") as f:
            pickle.dump(cur_index_performance, f)

        with open(paras_filename, "wb") as f:
            pickle.dump(cur_paras, f)

        return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self) -> np.ndarray:
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))

        cur_index_performance = self.default_index_performance.copy()

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        cur_c = self._raw_to_compact(np.copy(self.default_paras))
        best_c = np.copy(cur_c)
        init_state = np.concatenate((cur_c, best_c, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        return self.state_scaler.transform(init_state)

    def _get_next_state(
        self,
        cur_index_performance: np.ndarray,
        best_index_performance: np.ndarray,
        cur_paras: np.ndarray,
        best_paras: np.ndarray,
        target_rec: np.ndarray,
        num: int,
    ) -> np.ndarray:
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        cur_c = self._raw_to_compact(cur_paras)
        best_c = self._raw_to_compact(best_paras)
        next_state = np.concatenate((cur_c, best_c, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)
        return next_state

    def _step(self, actions: np.ndarray, data_feature: np.ndarray, performance_filename: str, paras_filename: str):
        self.steps += 1

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = self._wide_feature_block(cur_paras, data_feature)

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward(cur_index_performance)

        _ = self._record_best(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state_ = self.state_scaler.transform(next_state)

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta: np.ndarray) -> np.ndarray:
        reward_positive = (1 + delta) ** 2 - 1
        reward_negative = -(1 - delta) ** 2 + 1

        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0: np.ndarray, deltat: np.ndarray, deltat20: np.ndarray) -> np.ndarray:
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward(self, cur_index_performance: np.ndarray):
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)
        target_dec = st_counts_dec

        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        return reward, average_reward
