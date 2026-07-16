#!/usr/bin/env python3
"""
Zero-knowledge Latin Hypercube Sampling baseline for Milvus vector DB tuning.

Uniformly explores the multi-dimensional parameter space defined by a prior
config (or CLI knobs). Each sampled configuration is evaluated independently;
no historical knowledge or performance feedback is used between rounds.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_CONFIGURE = REPO_ROOT / "auto-configure"
PGTUNER_INTERFACE = REPO_ROOT / "pgtuner-configure" / "vdtuner_interface"

for p in (str(AUTO_CONFIGURE), str(PGTUNER_INTERFACE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_runner import run_milvus_benchmark  # noqa: E402
from configure import configure_index, configure_system, filter_index_rule, filter_system_rule  # noqa: E402
from lhs_sampler import iter_lhs_unique_assignments  # noqa: E402
from vdtuner_prior_space import load_prior_space  # noqa: E402


def _handle_signal(signum, _frame):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        signame = signal.Signals(signum).name
    except Exception:
        signame = str(signum)
    print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
    raise SystemExit(128 + int(signum))


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass


def _load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _inject_index_types(
    assignments: List[Dict[str, Any]],
    index_types: Sequence[str],
) -> None:
    if not index_types:
        return
    types = list(index_types)
    for i, assign in enumerate(assignments):
        assign["index_type"] = types[i % len(types)]


def _split_conf(full: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    index_names = {
        "index_type",
        "nlist",
        "nprobe",
        "m",
        "nbits",
        "M",
        "efConstruction",
        "ef",
        "reorder_k",
    }
    index_conf = {k: full[k] for k in index_names if k in full}
    system_conf = {k: v for k, v in full.items() if "*" in k}
    return index_conf, system_conf


def _evaluate_assignment(
    space,
    assignment: Mapping[str, Any],
    *,
    dataset: str,
    engine: str,
    process_tag: str,
    use_sudo: bool,
    timeout_sec: int,
) -> tuple[float, float, float, Dict[str, Any], Dict[str, Any]]:
    full = space.build_full_milvus_config(assignment)
    work = dict(full)
    configure_index(*filter_index_rule(work))
    system_conf = {k: full[k] for k in full if "*" in k}
    configure_system(filter_system_rule(system_conf))
    prec, rps, p95 = run_milvus_benchmark(
        engine,
        process_tag,
        dataset,
        use_sudo=use_sudo,
        timeout_sec=timeout_sec,
    )
    index_conf, _ = _split_conf(full)
    return prec, rps, p95, index_conf, system_conf


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Non-learning LHS baseline: uniformly sample the prior parameter space "
            "and benchmark each configuration on a target dataset."
        )
    )
    ap.add_argument(
        "--dataset",
        default="glove-100-angular",
        help="Benchmark dataset name (datasets.json key). Overridden by prior-config 'dataset'.",
    )
    ap.add_argument(
        "--iterations",
        "--rounds",
        dest="iterations",
        type=int,
        default=50,
        help="Number of LHS sampling rounds (benchmark evaluations).",
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed for Latin Hypercube Sampling.")
    ap.add_argument(
        "--prior-config",
        type=str,
        default=None,
        help="Optional prior JSON (tune_knobs, index_types, overrides, dataset, iterations, seed).",
    )
    ap.add_argument(
        "--index-types",
        nargs="*",
        default=None,
        help="Restrict index types when index_type is not in tune_knobs.",
    )
    ap.add_argument(
        "--tune-knobs",
        nargs="*",
        default=None,
        help="Restrict tunable knobs (defaults to all knobs when no prior-config).",
    )
    ap.add_argument(
        "--override-json",
        type=str,
        default=None,
        help="Inline JSON dict for knob range/enum overrides.",
    )
    ap.add_argument(
        "--record-log",
        type=str,
        default=None,
        help="Path for JSONL evaluation log (default: sampling-configure/logs/<dataset>.lhs.log).",
    )
    ap.add_argument("--engine", default="milvus-single-node")
    ap.add_argument("--process-tag", default="milvus-p10")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=2000)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampled assignments without running benchmarks.",
    )
    return ap.parse_args()


def main() -> None:
    _install_signal_handlers()
    args = _parse_args()

    priors: Dict[str, Any] = {}
    if args.prior_config:
        priors = _load_json(args.prior_config)

    dataset = args.dataset
    iterations = args.iterations
    seed = args.seed

    index_types: Optional[Sequence[str]] = priors.get("index_types", args.index_types)
    tune_knobs: Optional[Sequence[str]] = priors.get("tune_knobs", args.tune_knobs)

    overrides: Dict[str, Dict[str, Any]] = {}
    if isinstance(priors.get("overrides"), dict):
        overrides.update(priors["overrides"])
    if args.override_json:
        overrides.update(json.loads(args.override_json))

    if iterations <= 0:
        raise SystemExit("--iterations must be positive.")

    prior_path = Path(args.prior_config) if args.prior_config else None
    space = load_prior_space(
        prior_config_path=prior_path,
        index_types=index_types,
        tune_knobs=tune_knobs,
        overrides=overrides or None,
    )

    log_dir = Path(__file__).resolve().parent / "logs"
    record_log = Path(args.record_log) if args.record_log else log_dir / f"{dataset}.lhs.log"

    assignments = list(iter_lhs_unique_assignments(space, iterations, seed=seed))
    if "index_type" not in space.knob_names:
        effective_types = list(index_types) if index_types else None
        if effective_types is None:
            meta = space.knobs_meta.get("index_type", {})
            effective_types = list(meta.get("enum_values", []))
        _inject_index_types(assignments, effective_types or ["AUTOINDEX"])

    if len(assignments) < iterations:
        print(
            f"[lhs-baseline] warning: requested {iterations} unique samples but "
            f"only {len(assignments)} distinct configurations in the discrete space.",
            flush=True,
        )

    print(
        f"[lhs-baseline] dataset={dataset} rounds={len(assignments)} "
        f"compact_dim={space.compact_dim} seed={seed} log={record_log}",
        flush=True,
    )

    if args.dry_run:
        for i, assign in enumerate(assignments, 1):
            print(json.dumps({"round": i, "assignment": assign}, ensure_ascii=False))
        return

    record_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, assignment in enumerate(assignments, 1):
        print(f"[lhs-baseline] round {i}/{len(assignments)}", flush=True)
        try:
            prec, rps, p95, index_conf, system_conf = _evaluate_assignment(
                space,
                assignment,
                dataset=dataset,
                engine=args.engine,
                process_tag=args.process_tag,
                use_sudo=not args.no_sudo,
                timeout_sec=args.timeout_sec,
            )
        except Exception as e:
            print(f"[lhs-baseline] round {i} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            prec, rps, p95 = 0.1, 0.1, 0.1
            full = space.build_full_milvus_config(assignment)
            index_conf, system_conf = _split_conf(full)

        log_entry = {
            "iteration": i,
            "time": int(time.time() - t0),
            "index_conf": index_conf,
            "system_conf": system_conf,
            "precisions": prec,
            "p95time": p95,
            "RPS": rps,
            "method": "lhs_baseline",
            "seed": seed,
            "dataset": dataset,
        }
        with record_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(
            f"[lhs-baseline] round {i}/{len(assignments)} "
            f"prec={prec:.4f} rps={rps:.2f} p95={p95:.4f}",
            flush=True,
        )

    print(f"[lhs-baseline] finished {len(assignments)} evaluations; log={record_log}", flush=True)


if __name__ == "__main__":
    main()
