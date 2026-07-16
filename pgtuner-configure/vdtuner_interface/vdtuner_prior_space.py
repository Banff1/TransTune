# -*- coding: utf-8 -*-
"""
Decision space for PGTuner QPP + PCR aligned with ``auto-configure/vdtuner/main_tuner_priors.py``.

**Unified ``--prior-config`` JSON** (single file):

- Optional reserved keys: ``index_types``, ``tune_knobs``, ``overrides``, ``baseline`` (object: one value per
  tuned knob for PCR/eval default row). If ``baseline`` is omitted, defaults come from each knob's ``default``.
- Remaining top-level keys whose values look like Milvus knob specs (``"type": "integer"`` or ``"enum"``) are
  merged as knob metadata (same shape as ``whole_param.json``). If there are none, knobs load from
  ``whole_param.json`` (or ``--knob-json``).

- Produces:
  - **Wide** numeric vector for the query-performance predictor (QPP): one-hot enums, scaled integers.
  - **Compact** [0,1]^K actions for PCR (TD3), one scalar per tuned knob (VDTuner-style ``scale_back``).

Training CSV rows must contain one column per tuned knob (raw value: int / bool / str for ``index_type``).
Dataset tail columns stay the same as legacy ``read_data_new`` (SIZE … q_K_StdRatio).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

_VDB_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_KNOB_JSON = _VDB_ROOT / "auto-configure" / "whole_param.json"

# Integer knobs where we apply log10 before min–max scaling (matches legacy HNSW QPP head).
_LOG10_INT_NAMES = {
    "efConstruction",
    "ef",
    "nlist",
    "nprobe",
    "reorder_k",
    "dataCoord*segment*maxSize",
    "dataNode*segment*insertBufSize",
    "rootCoord*minSegmentSizeToEnableIndex",
    "common*gracefulTime",
}

DATASET_FEATURE_COLUMNS: List[str] = [
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

PERFORMANCE_COLUMNS: List[str] = ["recall", "average_construct_dc_counts", "average_search_dc_counts"]

# Top-level keys in a unified prior JSON (not Milvus knob specs). Inline knob entries are dicts with type integer/enum.
_RESERVED_PRIOR_KEYS = frozenset(
    {
        "baseline",
        "index_types",
        "tune_knobs",
        "overrides",
        "dataset",
        "iterations",
        "seed",
    }
)


def _split_unified_prior_json(raw: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Separate inline knob definitions from meta (baseline, tune_knobs, …)."""
    knobs: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in _RESERVED_PRIOR_KEYS:
            meta[k] = v
            continue
        if isinstance(v, dict) and v.get("type") in ("integer", "enum"):
            knobs[k] = v
        else:
            meta[k] = v
    return knobs, meta


def _apply_knob_overrides(knobs: Dict[str, Dict[str, Any]], overrides: Mapping[str, Mapping[str, Any]]) -> None:
    """Same rules as ``main_tuner_priors._apply_knob_overrides`` (subset: integer + enum)."""
    for knob_name, patch in overrides.items():
        if knob_name not in knobs:
            raise ValueError(f"Unknown knob in overrides: {knob_name}")
        cur = knobs[knob_name]
        ktype = cur.get("type")
        if "default" in patch:
            cur["default"] = patch["default"]
        if ktype == "integer":
            if "enum_values" in patch:
                raise ValueError(f"Knob {knob_name} is integer; enum_values override is invalid.")
            if "min" in patch:
                cur["min"] = patch["min"]
            if "max" in patch:
                cur["max"] = patch["max"]
            if "min" in cur and "max" in cur and cur["min"] > cur["max"]:
                raise ValueError(f"Invalid override for {knob_name}: min > max ({cur['min']} > {cur['max']})")
            if "default" in cur and "min" in cur and "max" in cur:
                if cur["default"] < cur["min"]:
                    cur["default"] = cur["min"]
                if cur["default"] > cur["max"]:
                    cur["default"] = cur["max"]
        elif ktype == "enum":
            if "min" in patch or "max" in patch:
                raise ValueError(f"Knob {knob_name} is enum; min/max override is invalid.")
            if "enum_values" in patch:
                vals = list(patch["enum_values"])
                if len(vals) == 0:
                    raise ValueError(f"Invalid override for {knob_name}: enum_values is empty.")
                cur["enum_values"] = vals
            if "enum_values" in cur and "default" in cur:
                if cur["default"] not in cur["enum_values"]:
                    cur["default"] = cur["enum_values"][0]
        else:
            raise ValueError(f"Unsupported knob type for {knob_name}: {ktype}")


@dataclass(frozen=True)
class KnobSlot:
    name: str
    kind: str  # "integer" | "enum"
    use_log10: bool
    int_min: int
    int_max: int
    enum_values: Tuple[Any, ...]


class VDTunerPriorSpace:
    """
    Tuned knobs (order = ``knob_names``). Wide width = sum(enum widths) + (#integer slots).
    """

    def __init__(self, knob_names: List[str], knobs_meta: Dict[str, Dict[str, Any]]):
        self.knob_names = list(knob_names)
        self.knobs_meta = knobs_meta
        self._slots: List[KnobSlot] = []
        self._wide_span: List[Tuple[int, int]] = []  # (start, end) per knob in wide vector
        pos = 0
        for name in self.knob_names:
            meta = knobs_meta[name]
            ktype = meta["type"]
            if ktype == "integer":
                use_log = name in _LOG10_INT_NAMES
                slot = KnobSlot(
                    name=name,
                    kind="integer",
                    use_log10=use_log,
                    int_min=int(meta["min"]),
                    int_max=int(meta["max"]),
                    enum_values=(),
                )
                self._slots.append(slot)
                self._wide_span.append((pos, pos + 1))
                pos += 1
            elif ktype == "enum":
                vals = tuple(meta["enum_values"])
                slot = KnobSlot(
                    name=name,
                    kind="enum",
                    use_log10=False,
                    int_min=0,
                    int_max=0,
                    enum_values=vals,
                )
                self._slots.append(slot)
                self._wide_span.append((pos, pos + len(vals)))
                pos += len(vals)
            else:
                raise ValueError(f"Unsupported type for knob {name}: {ktype}")
        self.wide_dim = pos
        self.compact_dim = len(self.knob_names)

    @property
    def qpp_input_dim(self) -> int:
        return self.wide_dim + len(DATASET_FEATURE_COLUMNS)

    def required_train_columns(self) -> List[str]:
        return ["FileName"] + self.knob_names + DATASET_FEATURE_COLUMNS + PERFORMANCE_COLUMNS

    def default_config_dict(self) -> Dict[str, Any]:
        return {name: self.knobs_meta[name]["default"] for name in self.knob_names}

    def factorial_space_size(self) -> int:
        """Cartesian product size: every integer in [min,max], every enum value (after priors)."""
        n = 1
        for name in self.knob_names:
            meta = self.knobs_meta[name]
            if meta["type"] == "integer":
                lo, hi = int(meta["min"]), int(meta["max"])
                w = hi - lo + 1
                if w <= 0:
                    raise ValueError(f"Invalid integer range for {name}: [{lo}, {hi}]")
                n *= w
            else:
                n *= len(meta["enum_values"])
        return int(n)

    def iter_factorial_assignments(self) -> Iterator[Dict[str, Any]]:
        import itertools

        domains: List = []
        for name in self.knob_names:
            meta = self.knobs_meta[name]
            if meta["type"] == "integer":
                lo, hi = int(meta["min"]), int(meta["max"])
                domains.append(range(lo, hi + 1))
            else:
                domains.append(list(meta["enum_values"]))
        for tup in itertools.product(*domains):
            yield dict(zip(self.knob_names, tup))

    def build_full_milvus_config(self, assignment: Mapping[str, Any]) -> Dict[str, Any]:
        """Full knob dict for ``configure`` (defaults + one assignment row over ``knob_names``)."""
        conf: Dict[str, Any] = {k: self.knobs_meta[k]["default"] for k in self.knobs_meta}
        for name in self.knob_names:
            conf[name] = assignment[name]
        return conf

    def scale_forward_compact(self, name: str, real_val: Any) -> float:
        """Map one raw knob value to [0, 1] (VDTuner KnobStand-style)."""
        meta = self.knobs_meta[name]
        if meta["type"] == "integer":
            lo, hi = int(meta["min"]), int(meta["max"])
            v = int(round(float(real_val)))
            v = max(lo, min(hi, v))
            return (v - lo) / (hi - lo) if hi > lo else 0.0
        if meta["type"] == "enum":
            vals = list(meta["enum_values"])
            if real_val not in vals:
                real_val = vals[0]
            idx = vals.index(real_val)
            n = len(vals)
            return idx / n if n > 0 else 0.0
        raise TypeError(meta["type"])

    def scale_back_compact(self, name: str, u: float) -> Any:
        u = float(max(0.0, min(1.0, u)))
        meta = self.knobs_meta[name]
        if meta["type"] == "integer":
            lo, hi = int(meta["min"]), int(meta["max"])
            if hi <= lo:
                return lo
            real = u * (hi - lo) + lo
            return int(round(real))
        if meta["type"] == "enum":
            vals = list(meta["enum_values"])
            n = len(vals)
            idx = int(n * u)
            idx = min(n - 1, max(0, idx))  # u==1.0 → idx n-1
            return vals[idx]
        raise TypeError(meta["type"])

    def row_dict_from_series(self, row: Any) -> Dict[str, Any]:
        return {name: row[name] for name in self.knob_names}

    def encode_wide_row(self, values: Mapping[str, Any]) -> "np.ndarray":
        import numpy as np

        out = np.zeros((self.wide_dim,), dtype=np.float32)
        for i, slot in enumerate(self._slots):
            s, e = self._wide_span[i]
            v = values[slot.name]
            meta = self.knobs_meta[slot.name]
            if slot.kind == "integer":
                lo, hi = slot.int_min, slot.int_max
                iv = int(round(float(v)))
                iv = max(lo, min(hi, iv))
                if slot.use_log10:
                    x = np.log10(float(iv))
                    xmin = np.log10(float(max(lo, 1)))
                    xmax = np.log10(float(max(hi, 1)))
                    out[s] = float((x - xmin) / (xmax - xmin)) if xmax > xmin else 0.0
                else:
                    out[s] = float((iv - lo) / (hi - lo)) if hi > lo else 0.0
            else:
                vals = list(meta["enum_values"])
                if v not in vals:
                    v = meta["default"]
                oh = np.zeros((e - s,), dtype=np.float32)
                oh[vals.index(v)] = 1.0
                out[s:e] = oh
        return out

    def wide_min_max_head(self) -> Tuple["np.ndarray", "np.ndarray"]:
        """Fixed [0,1] bounds for every wide dimension (after encoding)."""
        import numpy as np

        lo = np.zeros((self.wide_dim,), dtype=np.float32)
        hi = np.ones((self.wide_dim,), dtype=np.float32)
        return lo, hi

    def compact_array_from_dict(self, values: Mapping[str, Any]) -> "np.ndarray":
        import numpy as np

        return np.array([self.scale_forward_compact(n, values[n]) for n in self.knob_names], dtype=np.float32)

    def actions_to_raw(self, actions: "np.ndarray") -> "np.ndarray":
        """Map actor outputs in [0, 1]^K to stored rows: ints as floats, bool enums {0,1}, other enums as index."""
        import numpy as np

        out = np.zeros((actions.shape[0], self.compact_dim), dtype=np.float32)
        for i in range(actions.shape[0]):
            for j, name in enumerate(self.knob_names):
                raw = self.scale_back_compact(name, float(actions[i, j]))
                meta = self.knobs_meta[name]
                if meta["type"] == "integer":
                    out[i, j] = float(int(raw))
                else:
                    vals = list(meta["enum_values"])
                    if all(type(x) is bool for x in vals):
                        out[i, j] = 1.0 if bool(raw) else 0.0
                    else:
                        out[i, j] = float(vals.index(raw))
        return out

    def raw_numpy_row_to_value_dict(self, row: "np.ndarray") -> Dict[str, Any]:
        """Turn one stored raw row into values Milvus / ``encode_wide_row`` accepts."""
        d: Dict[str, Any] = {}
        for j, name in enumerate(self.knob_names):
            meta = self.knobs_meta[name]
            v = float(row[j])
            if meta["type"] == "integer":
                lo, hi = int(meta["min"]), int(meta["max"])
                d[name] = int(round(max(lo, min(hi, v))))
            else:
                vals = list(meta["enum_values"])
                if all(type(x) is bool for x in vals):
                    d[name] = bool(v >= 0.5)
                else:
                    idx = int(round(v))
                    d[name] = vals[max(0, min(len(vals) - 1, idx))]
        return d


def assignment_as_tuple(space: VDTunerPriorSpace, assignment: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Stable key for deduplication (matches resume semantics in ``milvus_qpp_data``)."""
    t: List[Any] = []
    for name in space.knob_names:
        meta = space.knobs_meta[name]
        v = assignment[name]
        if meta["type"] == "integer":
            t.append(int(round(float(v))))
        else:
            vals = list(meta["enum_values"])
            if all(type(x) is bool for x in vals):
                t.append(bool(v))
            else:
                t.append(v)
    return tuple(t)


def iter_lhs_unique_assignments(
    space: VDTunerPriorSpace,
    n_target: int,
    *,
    seed: int = 0,
    max_rounds: int = 500,
) -> Iterator[Dict[str, Any]]:
    """
    Latin Hypercube samples in [0,1]^K per knob, mapped with ``scale_back_compact`` (VDTuner-style).
    Yields at most ``n_target`` unique discrete configurations.
    """
    try:
        from scipy.stats import qmc
    except ImportError as e:  # pragma: no cover
        raise ImportError("LHS sampling requires scipy (see pgtuner-configure/requirements.txt)") from e

    import numpy as np

    k = space.compact_dim
    if k == 0 or n_target <= 0:
        return
    seen: set = set()
    round_idx = 0
    while len(seen) < n_target and round_idx < max_rounds:
        need = n_target - len(seen)
        batch = max(min(8192, need * 4), k * 4)
        sampler = qmc.LatinHypercube(d=k, seed=int(seed) + round_idx)
        u = np.asarray(sampler.random(n=batch), dtype=np.float64)
        np.clip(u, 0.0, 1.0 - 1e-15, out=u)
        round_idx += 1
        found_this_round = False
        for i in range(batch):
            assignment = {
                name: space.scale_back_compact(name, float(u[i, j])) for j, name in enumerate(space.knob_names)
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


@dataclass
class PriorConfigBundle:
    """Single ``--prior-config`` file: space + PCR/eval baseline row."""

    space: VDTunerPriorSpace
    baseline: Dict[str, Any]
    prior_config_path: Optional[Path] = None


def load_prior_bundle(
    knob_json: Optional[Path] = None,
    *,
    prior_config_path: Optional[Path] = None,
    index_types: Optional[Sequence[str]] = None,
    tune_knobs: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> PriorConfigBundle:
    """
    Load knob metadata + optional ``baseline`` from one JSON (see module doc).

    If ``prior_config_path`` contains inline knob specs (``type``: integer/enum), those define
    ``knobs_meta``; otherwise knobs load from ``knob_json`` or ``whole_param.json``.

    ``baseline`` in the JSON must assign every ``tune_knobs`` entry; missing keys fall back to
    knob ``default`` after overrides.
    """
    meta: Dict[str, Any] = {}
    knobs: Optional[Dict[str, Dict[str, Any]]] = None
    p_path = Path(prior_config_path) if prior_config_path is not None else None

    if p_path is not None:
        with p_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("prior-config JSON must be an object")
        knobs_inline, meta = _split_unified_prior_json(raw)
        if knobs_inline:
            knobs = knobs_inline

    if knobs is None:
        path = Path(knob_json) if knob_json else _DEFAULT_KNOB_JSON
        with path.open("r", encoding="utf-8") as f:
            knobs = json.load(f)

    merged_overrides: Dict[str, Dict[str, Any]] = {}
    if isinstance(meta.get("overrides"), dict):
        merged_overrides.update(meta["overrides"])
    if overrides:
        merged_overrides.update(dict(overrides))

    idx_types = index_types if index_types is not None else meta.get("index_types")
    t_knobs = tune_knobs if tune_knobs is not None else meta.get("tune_knobs")

    if merged_overrides:
        _apply_knob_overrides(knobs, merged_overrides)

    if idx_types is not None:
        _apply_knob_overrides(knobs, {"index_type": {"enum_values": list(idx_types)}})

    if t_knobs is None:
        names_all = list(knobs.keys())
    else:
        names_all = list(t_knobs)
        for n in names_all:
            if n not in knobs:
                raise ValueError(f"tune_knobs contains unknown knob: {n}")

    space = VDTunerPriorSpace(names_all, knobs)

    baseline_raw = meta.get("baseline")
    if baseline_raw is None:
        baseline = space.default_config_dict()
    elif isinstance(baseline_raw, dict):
        baseline = dict(baseline_raw)
    else:
        raise ValueError("baseline must be a JSON object")

    for name in space.knob_names:
        if name not in baseline:
            baseline[name] = space.knobs_meta[name]["default"]

    return PriorConfigBundle(space=space, baseline=baseline, prior_config_path=p_path)


def load_prior_space(
    knob_json: Optional[Path] = None,
    *,
    prior_config_path: Optional[Path] = None,
    index_types: Optional[Sequence[str]] = None,
    tune_knobs: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> VDTunerPriorSpace:
    """Return only the space; use ``load_prior_bundle`` when you need ``baseline``."""
    return load_prior_bundle(
        knob_json,
        prior_config_path=prior_config_path,
        index_types=index_types,
        tune_knobs=tune_knobs,
        overrides=overrides,
    ).space
