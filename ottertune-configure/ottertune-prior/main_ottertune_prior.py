#!/usr/bin/env python
"""
OtterTune with prior: warm-start GPRGD from historical JSON records.

Supports prior files like log/*_ottertune.json (fractional sealProportion, optional p95time,
null precisions/RPS skipped) and legacy flat logs (integer percent, no p95time).
Skips LHS initial design and benchmark runs for prior points.

Optimization target matches VDTuner (auto-configure/vdtuner/optimizer_pobo_sa.py reward_transform):
per index_type, nondominated sort and the same reference-point normalization for
(precision, RPS); GPRGD fits a scalar — we minimize -(min(norm_precision, norm_RPS)).
"""
import argparse
import json
import os
import subprocess as sp
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# vdb-tuning/ottertune-configure/ottertune-prior/this_file.py -> repo root
REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
OTTERTUNE_SERVER_PATH = os.path.join(REPO_ROOT, "ottertune-configure", "ottertune", "server")
VDTUNER_AUTO_CONFIGURE_PATH = os.path.join(REPO_ROOT, "auto-configure")
OTTERTUNE_CONFIGURE_PATH = os.path.join(REPO_ROOT, "ottertune-configure")

for _p in (OTTERTUNE_SERVER_PATH, VDTUNER_AUTO_CONFIGURE_PATH, OTTERTUNE_CONFIGURE_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.gp_tf import GPRGD
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv

KNOB_PATH = os.path.join(REPO_ROOT, "auto-configure", "whole_param.json")
RUN_ENGINE_PATH = os.path.join(REPO_ROOT, "vector-db-benchmark-master", "run_engine_test.sh")
DEFAULT_PRIOR_JSON = os.path.join(
    REPO_ROOT, "ottertune-configure", "log", "random-match-int-100-angular-no-filters_ottertune.json"
)
USE_SUDO = True


def _coerce_knob_real_value(name, value, knob):
    """Map JSON knob values to the types/ranges expected by whole_param.json / KnobStand."""
    if knob["type"] == "integer":
        v = float(value)
        # OtterTune log export: seal proportion as fraction in (0, 1], e.g. 0.88 -> 88
        # Log JSON often uses fractions (e.g. 0.88); legacy logs use integer percent (e.g. 88).
        # Use strict < 1 so 1.0 stays as 1 (1%), not 100.
        if name == "dataCoord*segment*sealProportion" and 0 < v < 1:
            v = int(round(v * 100))
        else:
            v = int(round(v))
        lo, hi = int(knob["min"]), int(knob["max"])
        return max(lo, min(hi, v))
    if knob["type"] == "enum":
        return value
    return value


def run_engine_test(dataset: str, use_sudo=None):
    """Run benchmark; return (rps, precision, p95_time) or (None, None, None). Same parse as main_ottertune."""
    if use_sudo is None:
        use_sudo = USE_SUDO
    try:
        cmd = ["timeout", "900", RUN_ENGINE_PATH, "milvus-single-node", "milvus-p10", dataset]
        if use_sudo:
            cmd = ["sudo"] + cmd
        result = sp.run(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=os.path.dirname(RUN_ENGINE_PATH),
        )
        result_output = result.stdout.decode(errors="ignore") + result.stderr.decode(errors="ignore")
        lines = result_output.strip().split("\n")

        numeric_values = []
        for line in reversed(lines):
            if "测试结果摘要" in line or "📊" in line or "结果" in line:
                break
            for word in line.strip().split():
                try:
                    numeric_values.append(float(word))
                except ValueError:
                    continue

        from_reversed = len(numeric_values) >= 3
        if not from_reversed:
            numeric_values = []
            for item in result_output.strip().split():
                try:
                    numeric_values.append(float(item))
                except ValueError:
                    continue

        if len(numeric_values) < 2:
            return None, None, None

        rps = numeric_values[-2]
        precision = (
            numeric_values[-1]
            if from_reversed
            else (numeric_values[-3] if len(numeric_values) >= 3 else numeric_values[-1])
        )
        if from_reversed and len(numeric_values) >= 3:
            p95 = float(numeric_values[-3])
        elif len(numeric_values) >= 3:
            p95 = float(numeric_values[-1])
        else:
            p95 = 0.0

        if 0 <= precision <= 1 and rps > 0:
            return rps, precision, p95
        return None, None, None
    except Exception as e:
        print(f"Error running engine test: {e}")
        return None, None, None


def fast_non_dominated_sort(P: np.ndarray):
    """
    Same 2-D dominance rule as auto-configure/vdtuner/optimizer_pobo_sa.py.
    P: (n, 2), larger is better on both axes.
    Returns (rank, fronts) where fronts[0] is the Pareto front indices.
    """

    def compare(p1, p2):
        p1_dom_p2 = True
        p2_dom_p1 = True
        for i in range(len(p1)):
            if p1[i] < p2[i]:
                p1_dom_p2 = False
            if p1[i] > p2[i]:
                p2_dom_p1 = False
        if p1_dom_p2 == p2_dom_p1:
            return 0
        return 1 if p1_dom_p2 else -1

    p_size = len(P)
    n_dom = np.zeros(p_size, dtype=int)
    s_list = []
    fronts = []
    rank = np.full(p_size, -1)

    f0 = []
    for p in range(p_size):
        n_p = 0
        s_p = []
        for q in range(p_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                s_p.append(q)
            elif cmp == -1:
                n_p += 1
        s_list.append(s_p)
        n_dom[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f0.append(p)
    fronts.append(f0)
    i = 0
    while len(fronts[i]) != 0:
        q_next = []
        for p in fronts[i]:
            for q in s_list[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    q_next.append(q)
        i += 1
        fronts.append(q_next)
    return rank, fronts


def _vdtuner_normalize_per_index_type(
    prec: np.ndarray, rps: np.ndarray, index_types: Sequence[str]
) -> np.ndarray:
    """
    Match PollingBayesianOptimization.reward_transform normalization:
    per index_type group, pick chosen_ref via the same fitness tie-break,
    then divide (precision, RPS) by chosen_ref (both larger-is-better).
    Returns (N, 2) normalized metrics.
    """
    prec = np.asarray(prec, dtype=np.float64).ravel()
    rps = np.asarray(rps, dtype=np.float64).ravel()
    n = prec.shape[0]
    norm = np.zeros((n, 2), dtype=np.float64)
    by_type: dict = {}
    for i in range(n):
        by_type.setdefault(index_types[i], []).append(i)
    for idxs in by_type.values():
        idx_arr = np.array(idxs, dtype=int)
        y_k = np.stack([prec[idx_arr], rps[idx_arr]], axis=1)
        _, popu = fast_non_dominated_sort(y_k)
        max0 = np.max(y_k[:, 0]) + 1e-12
        max1 = np.max(y_k[:, 1]) + 1e-12
        fitness = -1.0 / (np.abs(y_k[:, 0] / max0 - y_k[:, 1] / max1) + 1e-6)
        front = popu[0]
        fitness[front] = -fitness[front]
        chosen_idx = int(np.argmax(fitness))
        chosen_ref = y_k[chosen_idx, :]
        y_norm = y_k.copy()
        y_norm[:, 0] /= chosen_ref[0] + 1e-12
        y_norm[:, 1] /= chosen_ref[1] + 1e-12
        norm[idx_arr] = y_norm
    return norm


def scalar_objective_vdtuner_style(
    prec: np.ndarray, rps: np.ndarray, index_types: Sequence[str]
) -> np.ndarray:
    """
    VDTuner fits a 2-objective GP (precision, RPS); GPRGD is scalar, so we minimize
    -(min of normalized objectives) to seek Pareto-aligned trade-offs like EHVI.
    """
    norm = _vdtuner_normalize_per_index_type(prec, rps, index_types)
    m = np.minimum(norm[:, 0], norm[:, 1])
    return (-m).astype(np.float32).reshape(-1, 1)


def rescaled_y(y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    if y_max > y_min:
        y_scaled = (y_train - y_min) / (y_max - y_min)
    else:
        y_scaled = y_train.astype(np.float32)
    return y_scaled.astype(np.float32), y_min, y_max


def record_to_vector(record: dict, knob_stand: KnobStand, names: List[str]) -> Optional[np.ndarray]:
    merged = {}
    merged.update(record.get("index_conf", {}))
    merged.update(record.get("system_conf", {}))
    detail = knob_stand.knobs_detail
    vec = []
    try:
        for name in names:
            if name not in merged:
                return None
            real_val = _coerce_knob_real_value(name, merged[name], detail[name])
            vec.append(knob_stand.scale_forward(name, real_val))
        return np.array(vec, dtype=np.float32)
    except (ValueError, KeyError, TypeError):
        return None


def infer_top_level_key_order(records):
    """Preserve key order: first-seen across all rows (matches common JSON export layout)."""
    order = []
    seen = set()
    for r in records:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                order.append(k)
    return order


def system_conf_uses_fractional_seal_proportion(records):
    """True if any prior row stores dataCoord*segment*sealProportion as (0,1) fraction."""
    key = "dataCoord*segment*sealProportion"
    for r in records:
        sc = r.get("system_conf")
        if not isinstance(sc, dict) or key not in sc:
            continue
        v = sc[key]
        if isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 1:
            return True
    return False


def format_system_conf_for_output(system_conf, seal_as_fraction):
    """KnobStand uses integer percent; emit fraction in (0,1] when prior JSON did."""
    out = {}
    key = "dataCoord*segment*sealProportion"
    for k, v in system_conf.items():
        if k == key and seal_as_fraction:
            try:
                iv = int(v)
                out[k] = iv / 100.0
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def build_output_record(
    iteration,
    cumulative_time,
    index_conf,
    system_conf,
    precisions,
    p95_latency,
    rps,
    key_order,
    seal_as_fraction,
    elapsed_sec=None,
):
    """One object per run; top-level keys and order match prior file (key_order)."""
    index_out = dict(index_conf)
    system_out = format_system_conf_for_output(system_conf, seal_as_fraction)
    schema_has_p95 = "p95time" in key_order
    if schema_has_p95:
        t_time = int(round(elapsed_sec if elapsed_sec is not None else p95_latency))
    else:
        t_time = int(round(p95_latency))

    values_by_key = {
        "iteration": iteration,
        "time": cumulative_time,
        "index_conf": index_out,
        "system_conf": system_out,
        "precisions": precisions,
        "Time": t_time,
        "RPS": rps,
    }
    if schema_has_p95:
        values_by_key["p95time"] = float(p95_latency)

    out = {}
    for k in key_order:
        if k in values_by_key:
            out[k] = values_by_key[k]
    return out


def load_prior_from_json(
    path: str,
    knob_stand: KnobStand,
    names: List[str],
    log,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[str], List[dict]]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Prior JSON must be a JSON array, got {type(raw)}")

    xs, prec_list, rps_list, idx_types = [], [], [], []
    skipped = 0
    for i, rec in enumerate(raw):
        if rec.get("RPS") is None or rec.get("precisions") is None:
            skipped += 1
            log("Skip prior record {}: null RPS or precisions".format(i))
            continue
        v = record_to_vector(rec, knob_stand, names)
        if v is None:
            skipped += 1
            log("Skip prior record {}: missing keys or invalid enum/value".format(i))
            continue
        prec = float(rec.get("precisions", rec.get("precision", 0)))
        rps = float(rec["RPS"])
        ic = rec.get("index_conf") if isinstance(rec.get("index_conf"), dict) else {}
        idx = ic.get("index_type") if isinstance(ic, dict) else None
        if not idx:
            idx = "UNKNOWN"
        xs.append(v)
        prec_list.append(prec)
        rps_list.append(rps)
        idx_types.append(str(idx))

    if not xs:
        raise RuntimeError("No valid prior records loaded; check knob names and JSON schema.")

    if skipped:
        log(f"Loaded {len(xs)} prior points ({skipped} skipped).")
    else:
        log(f"Loaded {len(xs)} prior points.")

    X = np.stack(xs, axis=0)
    prec_a = np.array(prec_list, dtype=np.float64)
    rps_a = np.array(rps_list, dtype=np.float64)
    y = scalar_objective_vdtuner_style(prec_a, rps_a, idx_types)
    return X, y, prec_list, rps_list, idx_types, raw


def log_default(message, log_file=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")


def main():
    parser = argparse.ArgumentParser(description="OtterTune GPRGD warm-started from prior JSON.")
    parser.add_argument(
        "--prior-json",
        default=DEFAULT_PRIOR_JSON,
        help=(
            "Prior run log: JSON array of objects with index_conf, system_conf, precisions, RPS "
            "(see ottertune-configure/log/*_ottertune.json). sealProportion may be 0–1 fraction; "
            "integer knobs are clamped to whole_param ranges. Rows with null RPS/precisions are skipped for GP."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(_SCRIPT_DIR, "tuning_session.json"),
        help=(
            "Append-only timeline: prior rows unchanged, new rows use the same top-level key set/order "
            "and system_conf sealProportion style (fraction vs integer) as --prior-json."
        ),
    )
    parser.add_argument("--dataset", default="glove-100-angular", help="Benchmark dataset name.")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=243,
        help="BO iterations after prior (default 243 matches 250 total minus 7 LHS in main_ottertune).",
    )
    parser.add_argument("--log-file", default=None, help="Append human-readable log.")
    parser.add_argument(
        "--prec-threshold",
        type=float,
        default=0.93,
        help="Ignored (legacy). Objective matches VDTuner reward_transform; see module docstring.",
    )
    args = parser.parse_args()

    def log(msg):
        log_default(msg, args.log_file)

    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write("")

    log("--- OtterTune prior (GPRGD warm-start) ---")
    log(
        "Objective aligns with VDTuner: per index_type same normalization as reward_transform; "
        "scalar for GPRGD = -(min(norm_precision, norm_RPS)) (minimize)."
    )

    env = RealEnv(bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH, dataset=args.dataset)
    knob_stand = KnobStand(KNOB_PATH)
    num_knobs = len(env.names)
    log(f"Dataset={args.dataset}, knobs={num_knobs}")

    X_train, y_train, prec_hist, rps_hist, idx_hist, prior_records = load_prior_from_json(
        args.prior_json, knob_stand, env.names, log
    )
    prec_hist = list(prec_hist)
    rps_hist = list(rps_hist)
    idx_hist = list(idx_hist)
    y_train_scaled, _, _ = rescaled_y(y_train)
    log(f"Prior training shape: X={X_train.shape}, y={y_train.shape}")

    cumulative_time = max(int(r.get("time", 0) or 0) for r in prior_records)
    max_iter = max(int(r["iteration"]) for r in prior_records)
    output_key_order = infer_top_level_key_order(prior_records)
    seal_as_fraction = system_conf_uses_fractional_seal_proportion(prior_records)
    all_records = [json.loads(json.dumps(r)) for r in prior_records]
    log(
        "Output schema: keys={!r}, sealProportion_as_fraction={}".format(
            output_key_order, seal_as_fraction
        )
    )

    log("\n--- Initializing OtterTune GPRGD (no LHS / no initial benchmarks) ---\n")
    model = GPRGD(
        length_scale=1.0,
        magnitude=1.0,
        ridge=0.1,
        max_iter=20,
        learning_rate=0.001,
        check_numerics=True,
        debug=True,
    )

    X_min = np.zeros(num_knobs, dtype=np.float32)
    X_max = np.ones(num_knobs, dtype=np.float32)

    log("\n--- Iterative tuning ---\n")
    for it in range(args.num_iterations):
        log(f"\n--- Iteration {it + 1}/{args.num_iterations} (session) ---")

        y_train_scaled, y_min, y_max = rescaled_y(y_train)

        log("Fitting GPRGD model...")
        model.fit(X_train, y_train_scaled, X_min=X_min, X_max=X_max)

        num_candidates = 10
        X_candidates = LHS_sample(num_knobs, num_candidates, seed=it + 2)
        results = model.predict(X_candidates)
        minl_flat = np.asarray(results.minl).ravel()
        best_idx = int(np.argmin(minl_flat))
        best_candidate = results.minl_conf[best_idx]

        conf_values = [knob_stand.scale_back(env.names[j], best_candidate[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))

        log(f"Best candidate index_conf={index_conf}")
        log(f"Best candidate system_conf={system_conf}")

        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

        t_run = time.time()
        rps, precision, p95 = run_engine_test(args.dataset)
        elapsed = max(0, int(time.time() - t_run))
        cumulative_time += elapsed

        next_iteration = max_iter + it + 1

        if rps is not None and precision is not None:
            prec_hist.append(float(precision))
            rps_hist.append(float(rps))
            idx_hist.append(str(index_conf.get("index_type", "UNKNOWN")))
            X_train = np.vstack((X_train, best_candidate.reshape(1, -1)))
            y_train = scalar_objective_vdtuner_style(
                np.array(prec_hist, dtype=np.float64),
                np.array(rps_hist, dtype=np.float64),
                idx_hist,
            )
            objective = float(y_train[-1, 0])

            rec = build_output_record(
                next_iteration,
                cumulative_time,
                index_conf,
                system_conf,
                float(precision),
                p95,
                float(rps),
                output_key_order,
                seal_as_fraction,
                elapsed_sec=elapsed,
            )
            all_records.append(rec)
            with open(args.output_json, "w") as f:
                json.dump(all_records, f, indent=2)
            if "p95time" in output_key_order:
                log(
                    "Performance: RPS={:.4f}, precisions={:.4f}, p95time={:.6f}, Time(wall)={}, "
                    "objective={:.4f}, wrote {}".format(
                        rps, precision, p95, elapsed, objective, args.output_json
                    )
                )
            else:
                log(
                    "Performance: RPS={:.4f}, precisions={:.4f}, Time(p95)={}, "
                    "objective={:.4f}, wrote {}".format(
                        rps, precision, int(round(p95)), objective, args.output_json
                    )
                )
        else:
            log("Engine test failed; not appending JSON record.")

    log("\n--- Done ---")
    log(f"Final X_train shape: {X_train.shape}")


if __name__ == "__main__":
    main()
