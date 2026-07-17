import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.append("..")

from optimizer_pobo_sa import PollingBayesianOptimization
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
            # Some signals may not be available/assignable on all platforms.
            pass


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_knob_overrides(env: RealEnv, overrides: Dict[str, Dict[str, Any]]) -> None:
    """
    overrides example:
      {
        "index_type": {"enum_values": ["HNSW"]},
        "M": {"min": 8, "max": 48},
        "queryCoord*autoBalance": {"enum_values": [true]}
      }
    """
    knobs = env.knob_stand.knobs_detail

    for knob_name, patch in overrides.items():
        if knob_name not in knobs:
            raise ValueError(f"Unknown knob in overrides: {knob_name}")

        cur = knobs[knob_name]
        ktype = cur.get("type")

        # Allow overriding default for both integer/enum knobs.
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

            # Ensure default is in range.
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

            # Ensure default exists in enum values.
            if "enum_values" in cur and "default" in cur:
                if cur["default"] not in cur["enum_values"]:
                    cur["default"] = cur["enum_values"][0]
        else:
            raise ValueError(f"Unsupported knob type for {knob_name}: {ktype}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="VDTuner Bayesian tuning with priors (index types, knobs, and ranges)."
    )
    ap.add_argument("--dataset", default="glove-100-angular", help="Benchmark dataset name.")
    ap.add_argument("--iterations", type=int, default=200, help="Number of BO steps after init sampling.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed.")

    # Priors: either from config file, or from CLI.
    ap.add_argument(
        "--prior-config",
        type=str,
        default="prior/glove-25-angular copy.json",
        help="Path to a JSON config file with priors (index_types/tune_knobs/overrides).",
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
        help="Restrict tunable knobs by name, e.g. --tune-knobs M efConstruction ef dataCoord*segment*maxSize",
    )
    ap.add_argument(
        "--override-json",
        type=str,
        default=None,
        help='Inline JSON dict to override knob ranges/enums, e.g. \'{"M":{"min":8,"max":48},"index_type":{"enum_values":["HNSW"]}}\'',
    )
    return ap.parse_args()


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

    # prepare the environment
    env = RealEnv(dataset=dataset)

    # apply range/enum overrides BEFORE creating the optimizer (so default_conf uses updated knobs)
    if overrides:
        _apply_knob_overrides(env, overrides)

    model = PollingBayesianOptimization(
        env,
        seed=seed,
        allowed_index_types=index_types,
        tune_knobs=tune_knobs,
    )

    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for _i in range(iterations):
        model.step()


if __name__ == "__main__":
    main()

