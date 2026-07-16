"""Latin Hypercube Sampling helpers (stdlib-only; no scipy/numpy required)."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Mapping, Sequence


def latin_hypercube_sample(dimension: int, num_points: int, seed: int = 0) -> List[List[float]]:
    """
    Draw ``num_points`` samples in [0, 1]^``dimension`` using Latin Hypercube Sampling.

    Each dimension is stratified into ``num_points`` equal intervals; one sample is
    drawn uniformly from each interval, then columns are independently permuted.
    """
    if dimension <= 0 or num_points <= 0:
        return []

    rng = random.Random(int(seed))
    samples: List[List[float]] = [[0.0] * dimension for _ in range(num_points)]

    for j in range(dimension):
        strata = [(i + rng.random()) / num_points for i in range(num_points)]
        rng.shuffle(strata)
        for i in range(num_points):
            samples[i][j] = strata[i]

    return samples


def assignment_as_tuple(space, assignment: Mapping[str, Any]) -> tuple:
    t: List[Any] = []
    for name in space.knob_names:
        meta = space.knobs_meta[name]
        v = assignment[name]
        if meta["type"] == "integer":
            t.append(int(round(float(v))))
        else:
            vals = list(meta["enum_values"])
            if vals and all(type(x) is bool for x in vals):
                t.append(bool(v))
            else:
                t.append(v)
    return tuple(t)


def iter_lhs_unique_assignments(
    space,
    n_target: int,
    *,
    seed: int = 0,
    max_rounds: int = 500,
) -> Iterator[Dict[str, Any]]:
    """
    Latin Hypercube samples in [0,1]^K per knob, mapped with ``scale_back_compact``.
    Yields at most ``n_target`` unique discrete configurations.
    """
    k = space.compact_dim
    if k == 0 or n_target <= 0:
        return

    seen: set = set()
    round_idx = 0
    while len(seen) < n_target and round_idx < max_rounds:
        need = n_target - len(seen)
        batch = max(min(8192, need * 4), k * 4)
        u = latin_hypercube_sample(k, batch, seed=int(seed) + round_idx)
        round_idx += 1
        found_this_round = False
        for row in u:
            assignment = {
                name: space.scale_back_compact(name, float(row[j]))
                for j, name in enumerate(space.knob_names)
            }
            key = assignment_as_tuple(space, assignment)
            if key not in seen:
                seen.add(key)
                found_this_round = True
                yield assignment
                if len(seen) >= n_target:
                    return
        if not found_this_round:
            break
