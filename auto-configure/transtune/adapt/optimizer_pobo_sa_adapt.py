from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np


class PollingBayesianOptimization:
    """
    Lightweight optimizer for adapt workflow.

    This class intentionally avoids the original multi-index polling / successive-abandon
    logic because Strategy 2 in adapt mode fixes a single index type.
    """

    def __init__(
        self,
        env,
        seed: int = 1206,
        threshold=None,
        allowed_index_types: Optional[Sequence[str]] = None,
        tune_knobs: Optional[Sequence[str]] = None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.threshold = threshold
        self.knob_num = len(env.names)
        self.default_conf = list(self.env.default_conf())
        self.name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.env.names)}

        random.seed(seed)
        np.random.seed(seed)

        if "index_type" not in self.name_to_idx:
            raise ValueError("index_type knob is required in environment.")

        self.index_type_idx = self.name_to_idx["index_type"]
        if allowed_index_types:
            self.index_type = allowed_index_types[0]
        else:
            enum_values = self.env.knob_stand.knobs_detail["index_type"]["enum_values"]
            self.index_type = enum_values[0]

        self.fixed_index_val = self.env.knob_stand.scale_forward("index_type", self.index_type)

        if tune_knobs is None:
            self.tune_indices = [i for i, n in enumerate(self.env.names) if n != "index_type"]
        else:
            unknown = [k for k in tune_knobs if k not in self.name_to_idx]
            if unknown:
                raise ValueError(f"Unknown knob names in tune_knobs: {sorted(unknown)}")
            self.tune_indices = [
                self.name_to_idx[k]
                for k in tune_knobs
                if k != "index_type"
            ]

        self.X = {self.index_type: []}
        self.Y = {self.index_type: []}
        self._visited = set()

    def _score(self, y_row: Sequence[float]) -> float:
        # y_row layout from RealEnv: [precision, rps, time]
        if len(y_row) < 2:
            return -1.0
        p = float(y_row[0])
        rps = float(y_row[1])
        return p * max(rps, 0.0)

    def _clip01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _build_base_vector(self) -> np.ndarray:
        x = np.array(self.default_conf, dtype=float)
        x[self.index_type_idx] = self.fixed_index_val
        return x

    def _random_candidate(self) -> np.ndarray:
        x = self._build_base_vector()
        for i in self.tune_indices:
            x[i] = random.random()
        return x

    def _heuristic_candidate(self) -> np.ndarray:
        xs = self.X[self.index_type]
        ys = self.Y[self.index_type]
        if len(xs) < 5:
            return self._random_candidate()

        scores = np.array([self._score(y) for y in ys], dtype=float)
        order = np.argsort(-scores)
        top_k = min(5, len(order))
        anchor_idx = int(random.choice(order[:top_k]))
        anchor = np.array(xs[anchor_idx], dtype=float)

        # Anneal exploration as samples grow.
        sigma = max(0.04, 0.18 / np.sqrt(len(xs)))
        candidate = self._build_base_vector()
        candidate[:] = anchor
        candidate[self.index_type_idx] = self.fixed_index_val

        for i in self.tune_indices:
            noise = np.random.normal(0.0, sigma)
            candidate[i] = self._clip01(float(anchor[i] + noise))
        return candidate

    def _unique_candidate(self, max_retry: int = 20) -> np.ndarray:
        for _ in range(max_retry):
            cand = self._heuristic_candidate()
            key = tuple(np.round(cand, 5).tolist())
            if key not in self._visited:
                self._visited.add(key)
                return cand
        # Fallback if space is saturated under rounding.
        cand = self._random_candidate()
        self._visited.add(tuple(np.round(cand, 5).tolist()))
        return cand

    def _evaluate_and_record(self, x_vec: np.ndarray) -> None:
        y = self.env.get_state([x_vec.tolist()])
        y_row = np.asarray(y).tolist()[0]
        self.X[self.index_type].append(x_vec.tolist())
        self.Y[self.index_type].append(y_row)

    def init_sample(self):
        x0 = self._build_base_vector()
        self._visited.add(tuple(np.round(x0, 5).tolist()))
        self._evaluate_and_record(x0)

    def step(self):
        x_new = self._unique_candidate()
        self._evaluate_and_record(x_new)

