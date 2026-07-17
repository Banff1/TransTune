import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.append("..")

from optimizer_pobo_sa_constrained import ConstrainedPollingBayesianOptimization
from utils import RealEnv


def _handle_signal(signum, _frame):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        signame = signal.Signals(signum).name
    except Exception:
        signame = str(signum)
    print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
    raise SystemExit(128 + int(signum))


def _install_signal_handlers():
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            pass


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_knob_overrides(env: RealEnv, overrides: Dict[str, Dict[str, Any]]) -> None:
    knobs = env.knob_stand.knobs_detail

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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="VDTuner constrained BO with priors: maximize RPS subject to precision >= threshold."
    )
    ap.add_argument("--dataset", default="glove-100-angular", help="Benchmark dataset name.")
    ap.add_argument("--iterations", type=int, default=200, help="Number of BO steps after init sampling.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed.")

    ap.add_argument(
        "--prior-config",
        type=str,
        default="prior/glove-100-angular copy.json",
        help="Path to JSON config with priors (dataset/iterations/seed/index_types/tune_knobs/overrides).",
    )
    ap.add_argument(
        "--index-types",
        nargs="*",
        default=None,
        help="Restrict index types, e.g. --index-types HNSW IVF_PQ",
    )
    ap.add_argument(
        "--tune-knobs",
        nargs="*",
        default=None,
        help="Restrict tunable knobs by name, e.g. --tune-knobs M efConstruction ef",
    )
    ap.add_argument(
        "--override-json",
        type=str,
        default=None,
        help='Inline JSON dict to override knob ranges/enums, e.g. \'{"M":{"min":8,"max":48}}\'',
    )
    ap.add_argument(
        "--precision-thresholds",
        nargs="+",
        type=float,
        default=[0.90, 0.95, 0.99],
        help="Constraint thresholds. One tuning run per threshold.",
    )
    ap.add_argument(
        "--record-dir",
        type=str,
        default="record-constrained",
        help="Directory for per-threshold record logs.",
    )
    return ap.parse_args()


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def _run_single_threshold(
    *,
    dataset: str,
    iterations: int,
    seed: int,
    threshold: float,
    index_types: Optional[Sequence[str]],
    tune_knobs: Optional[Sequence[str]],
    overrides: Dict[str, Dict[str, Any]],
    record_dir: Path,
) -> None:
    tag = _threshold_tag(threshold)
    record_log_path = record_dir / f"{dataset}.precision_ge_{tag}.record.log"
    print(f"\n===== Start constrained run: precision >= {threshold:.2f} =====", flush=True)
    print(f"record_log: {record_log_path}", flush=True)

    env = RealEnv(dataset=dataset, record_log_path=str(record_log_path))
    if overrides:
        _apply_knob_overrides(env, overrides)

    model = ConstrainedPollingBayesianOptimization(
        env,
        seed=seed,
        precision_threshold=threshold,
        allowed_index_types=index_types,
        tune_knobs=tune_knobs,
    )
    model.init_sample()
    for _i in range(iterations):
        model.step()


def main():
    _install_signal_handlers()
    args = _parse_args()

    priors: Dict[str, Any] = {}
    if args.prior_config:
        priors = _load_json(args.prior_config)

    dataset = priors.get("dataset", args.dataset)
    iterations = int(priors.get("iterations", args.iterations))
    seed = int(priors.get("seed", args.seed))
    index_types: Optional[Sequence[str]] = priors.get("index_types", args.index_types)
    tune_knobs: Optional[Sequence[str]] = priors.get("tune_knobs", args.tune_knobs)

    overrides: Dict[str, Dict[str, Any]] = {}
    if isinstance(priors.get("overrides"), dict):
        overrides.update(priors["overrides"])
    if args.override_json:
        overrides.update(json.loads(args.override_json))

    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    for threshold in args.precision_thresholds:
        _run_single_threshold(
            dataset=dataset,
            iterations=iterations,
            seed=seed,
            threshold=float(threshold),
            index_types=index_types,
            tune_knobs=tune_knobs,
            overrides=overrides,
            record_dir=record_dir,
        )


if __name__ == "__main__":
    main()
