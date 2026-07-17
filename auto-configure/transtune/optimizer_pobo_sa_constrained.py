import numpy as np

from optimizer_pobo_sa import (
    PollingBayesianOptimization,
    fast_non_dominated_sort,
    hypervolume_calcu,
)


class ConstrainedPollingBayesianOptimization(PollingBayesianOptimization):
    """
    Constrained BO variant:
    - Hard feasibility preference: precision >= threshold
    - Objective: maximize RPS under that precision constraint
    """

    def __init__(
        self,
        env,
        seed: int = 1206,
        precision_threshold: float = 0.90,
        allowed_index_types=None,
        tune_knobs=None,
    ) -> None:
        super().__init__(
            env=env,
            seed=seed,
            threshold=precision_threshold,
            allowed_index_types=allowed_index_types,
            tune_knobs=tune_knobs,
        )
        self.precision_threshold = float(precision_threshold)

    def _transform_precision_rps(self, precision: float, rps: float):
        """
        Transform raw (precision, rps) to constrained objectives:
        - objective_0: feasibility score in [0, 1], equals 1 when precision >= threshold
        - objective_1: RPS if feasible else 0
        """
        thr = self.precision_threshold
        if precision >= thr:
            return 1.0, float(rps)
        soft_feasibility = max(0.0, min(1.0, float(precision) / (thr + 1e-12)))
        return soft_feasibility, 0.0

    def reward_transform(self):
        # Build transformed objective set per type.
        Y = []
        self.chosen_ref_k = dict.fromkeys(self.polling_index.keys(), None)
        for k, Y_k in self.Y.items():
            Y_k_arr = np.array(Y_k)[:, [0, 1]]  # raw: [precision, RPS]

            transformed = np.array(
                [self._transform_precision_rps(float(p), float(r)) for p, r in Y_k_arr]
            )

            _, popu = fast_non_dominated_sort(transformed)

            max0 = np.max(transformed[:, 0]) + 1e-12
            max1 = np.max(transformed[:, 1]) + 1e-12
            fitness = -1 / (
                np.abs(transformed[:, 0] / max0 - transformed[:, 1] / max1) + 1e-6
            )
            fitness[popu[0]] = -fitness[popu[0]]

            chosen_idx = np.argmax(fitness)
            chosen_ref = transformed[chosen_idx, :]
            self.chosen_ref_k[k] = chosen_ref.tolist()

            transformed_normalized = transformed.copy()
            transformed_normalized[:, 0] /= (chosen_ref[0] + 1e-12)
            transformed_normalized[:, 1] /= (chosen_ref[1] + 1e-12)
            Y += transformed_normalized.tolist()

        self.norm_X = [j for item in self.X.values() for j in item]
        self.norm_Y = Y

    def index_type_score(self):
        # Score index type contributions under transformed constrained objectives.
        Y = [j for item in self.Y.values() for j in item]
        Y_arr_raw = np.array(Y)[:, [0, 1]]
        Y_arr = np.array(
            [self._transform_precision_rps(float(p), float(r)) for p, r in Y_arr_raw]
        )

        _, popu = fast_non_dominated_sort(Y_arr)

        max0 = np.max(Y_arr[:, 0]) + 1e-12
        max1 = np.max(Y_arr[:, 1]) + 1e-12
        fitness = -1 / (np.abs(Y_arr[:, 0] / max0 - Y_arr[:, 1] / max1) + 1e-6)
        fitness[popu[0]] = -fitness[popu[0]]

        chosen_idx = np.argmax(fitness)
        self.chosen_ref_whole = Y_arr[chosen_idx, :]

        self.delta_hv = dict.fromkeys(self.remain_types, -9999)

        for k in self.remain_types:
            Y_nok = [j for i, item in self.Y.items() if i != k for j in item]
            Y_nok_raw = np.array(Y_nok)[:, [0, 1]]
            Y_nok_arr = np.array(
                [self._transform_precision_rps(float(p), float(r)) for p, r in Y_nok_raw]
            )

            Y_nok_arr_normalized = Y_nok_arr / (self.chosen_ref_whole + 1e-12)
            _, popu_nok = fast_non_dominated_sort(Y_nok_arr_normalized)
            popu0_nok = Y_nok_arr_normalized[popu_nok[0], :]
            self.delta_hv[k] = hypervolume_calcu(popu0_nok, ref_point=[0.5, 0.5])

        self.worst_type_record.append(max(self.delta_hv, key=lambda key: self.delta_hv[key]))
