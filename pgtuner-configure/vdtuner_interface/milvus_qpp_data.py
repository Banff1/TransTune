# -*- coding: utf-8 -*-
"""Build PGTuner QPP training CSV from Milvus benchmark grid + data_feature.csv."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PGTUNER_ROOT = Path(__file__).resolve().parent.parent


def _pgtuner_grids(which: str) -> Tuple[List[int], List[int], List[int]]:
    if which == "tiny":
        efcs = [20, 40, 80, 120, 200]
        ms = [4, 8, 16, 32]
        efs = [100, 200, 400, 600]
        return efcs, ms, efs
    if which == "medium":
        efcs = [20, 40, 60, 100, 160, 240, 360]
        ms = [4, 8, 12, 16, 24, 32]
        efs = [100, 200, 350, 500, 700, 900]
        return efcs, ms, efs
    # paper-style dense grid (subset of original PGTuner efC×M sweep + several ef)
    efcs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300]
    ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    efs = [100, 150, 200, 300, 400, 500, 600, 800, 1000]
    return efcs, ms, efs


def _valid_pairs(efcs: Sequence[int], ms: Sequence[int]) -> List[Tuple[int, int]]:
    out = []
    for efc in efcs:
        for m in ms:
            if m <= efc:
                out.append((efc, m))
            else:
                break
    return out


def collect_grid_rows_milvus(
    pgtuner_filename: str,
    sizes_csv: Path,
    out_csv: Path,
    grid: str,
    engine: str,
    process_tag: str,
    benchmark_dataset: str,
    use_sudo: bool,
    timeout_sec: int,
    resume: bool,
) -> None:
    """Run benchmark for each (efC,M,ef) and append rows to out_csv."""
    sys_path_hack = Path(__file__).resolve().parent
    if str(sys_path_hack) not in sys.path:
        sys.path.insert(0, str(sys_path_hack))
    from benchmark_runner import run_milvus_benchmark
    from hnsw_vdtuner_env import apply_hnsw_triplet

    sizes = pd.read_csv(sizes_csv)
    if pgtuner_filename not in set(sizes["FileName"].astype(str)):
        raise ValueError(f"Add {pgtuner_filename} to {sizes_csv} columns FileName,N,q_queries")

    row = sizes[sizes["FileName"] == pgtuner_filename].iloc[0]
    n_base = float(row["N"])
    q_queries = float(row["q_queries"])

    efcs, ms, efs = _pgtuner_grids(grid)
    pairs = _valid_pairs(efcs, ms)

    header = (
        "FileName,efConstruction,M,efSearch,construction_time,memory,recall,search_time,"
        "construct_dc_counts,search_dc_counts\n"
    )

    exist: set = set()
    if resume and out_csv.is_file():
        old = pd.read_csv(out_csv)
        if len(old):
            exist = set(zip(old["FileName"], old["efConstruction"], old["M"], old["efSearch"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_csv.is_file():
        out_csv.write_text(header, encoding="utf-8")

    for efc, m in pairs:
        for ef in efs:
            key = (pgtuner_filename, int(efc), int(m), int(ef))
            if key in exist:
                continue
            try:
                apply_hnsw_triplet(int(m), int(efc), int(ef))
                prec, _rps, p95 = run_milvus_benchmark(
                    engine, process_tag, benchmark_dataset, use_sudo=use_sudo, timeout_sec=timeout_sec
                )
            except Exception as e:
                print(
                    f"[collect] skip {key}: benchmark or configure failed ({type(e).__name__}: {e})",
                    flush=True,
                )
                traceback.print_exc()
                continue

            search_time = float(p95)
            recall = float(prec)
            construct_dc_counts = max(1, int(m * efc * np.sqrt(max(n_base, 1.0))))
            search_dc_counts = max(1, int(search_time * 1e6 * max(q_queries, 1.0)))

            line = (
                f"{pgtuner_filename},{int(efc)},{int(m)},{int(ef)},"
                f"0.0,0,{recall},{search_time},{construct_dc_counts},{search_dc_counts}\n"
            )
            with open(out_csv, "a", encoding="utf-8") as fp:
                fp.write(line)
            exist.add(key)
            print(f"logged {key} recall={recall} p95={p95}", flush=True)


def merge_milvus_perf_to_train_csv(
    data_feature_csv: Path,
    milvus_perf_csv: Path,
    sizes_csv: Path,
    output_train_csv: Path,
    file_name: Optional[str] = None,
) -> None:
    """Produce train_data.csv-style table for query_performance_predict/train.py."""
    df_feat = pd.read_csv(data_feature_csv)
    df_perf = pd.read_csv(milvus_perf_csv)
    if file_name is not None:
        df_perf = df_perf[df_perf["FileName"] == file_name]
    sizes = pd.read_csv(sizes_csv).set_index("FileName")

    def apply_avg(row: pd.Series) -> pd.Series:
        fn = str(row["FileName"])
        n = float(sizes.loc[fn, "N"])
        q = float(sizes.loc[fn, "q_queries"])
        row["average_construct_dc_counts"] = int(row["construct_dc_counts"] / n + 1)
        row["average_search_dc_counts"] = int(row["search_dc_counts"] / q + 1)
        row["whole_search_time"] = row["search_time"] * q
        return row

    df_perf = df_perf.apply(apply_avg, axis=1)
    merged = pd.merge(df_feat, df_perf, on="FileName", how="right")
    merged.to_csv(output_train_csv, index=False)
    print(f"Wrote {output_train_csv} ({len(merged)} rows)")


def _estimate_construct_dc_counts(conf: dict, n_base: float) -> int:
    it = conf.get("index_type")
    if it == "HNSW":
        m = int(conf.get("M", 32))
        efc = int(conf.get("efConstruction", 200))
        return max(1, int(m * efc * float(np.sqrt(max(n_base, 1.0)))))
    if it in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
        nlist = int(conf.get("nlist", 128))
        npr_pct = int(conf.get("nprobe", 8))
        return max(1, int(nlist * max(1, npr_pct) * float(np.sqrt(max(n_base, 1.0))) / 50))
    if it == "SCANN":
        rk = int(conf.get("reorder_k", 500))
        return max(1, int(rk * float(np.sqrt(max(n_base, 1.0))) / 100))
    return max(1, int(float(np.sqrt(max(n_base, 1.0))) * 1000))


def _normalize_resume_value(name: str, raw: Any, knobs_meta: dict) -> Any:
    meta = knobs_meta[name]
    if meta["type"] == "integer":
        return int(raw)
    if meta["type"] == "enum":
        vals = list(meta["enum_values"])
        if all(type(x) is bool for x in vals):
            if isinstance(raw, str):
                return raw.lower() in ("true", "1", "yes")
            return bool(raw)
        return raw
    return raw


def _row_resume_key(pgtuner_filename: str, knob_names: List[str], knobs_meta: dict, assignment: dict) -> tuple:
    return (pgtuner_filename,) + tuple(_normalize_resume_value(k, assignment[k], knobs_meta) for k in knob_names)


def collect_prior_factorial_milvus(
    space: Any,
    pgtuner_filename: str,
    sizes_csv: Path,
    out_csv: Path,
    benchmark_dataset: str,
    engine: str,
    process_tag: str,
    use_sudo: bool,
    timeout_sec: int,
    resume: bool,
    max_configurations: int,
    sampling: str = "auto",
    lhs_seed: int = 42,
    baseline: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Collect Milvus benchmarks over the prior decision space.

    - ``sampling=exact``: full Cartesian product (must have ``factorial_size <= max_configurations``).
    - ``sampling=lhs``: at most ``max_configurations`` unique points via Latin Hypercube in [0,1]^K.
    - ``sampling=auto``: exact factorial if it fits under the cap, otherwise LHS with ``max_configurations`` samples.
    """
    sys_path_hack = Path(__file__).resolve().parent
    if str(sys_path_hack) not in sys.path:
        sys.path.insert(0, str(sys_path_hack))
    from benchmark_runner import run_milvus_benchmark

    vdb_tuning_root = PGTUNER_ROOT.parent
    auto_cfg = vdb_tuning_root / "auto-configure"
    if str(auto_cfg) not in sys.path:
        sys.path.insert(0, str(auto_cfg))
    from configure import configure_index, configure_system, filter_index_rule, filter_system_rule

    from vdtuner_prior_space import iter_lhs_unique_assignments

    nf = int(space.factorial_space_size())
    sampling = (sampling or "auto").lower()
    if sampling not in ("auto", "exact", "lhs"):
        raise SystemExit(f"Unknown --sampling {sampling}; use auto, exact, or lhs")

    if sampling == "exact":
        if nf > max_configurations:
            raise SystemExit(
                f"Factorial space has {nf} configurations (cap {max_configurations}). "
                f"Use --sampling auto or lhs, or narrow tune_knobs / overrides."
            )
        assignment_iter = space.iter_factorial_assignments()
        total_plan = nf
        use_lhs = False
    elif sampling == "lhs":
        print(
            f"[collect-prior] LHS sampling: target {max_configurations} unique configs (seed={lhs_seed}); "
            f"full-factorial count={nf}",
            flush=True,
        )
        assignment_iter = iter_lhs_unique_assignments(space, max_configurations, seed=lhs_seed)
        total_plan = max_configurations
        use_lhs = True
    else:
        if nf <= max_configurations:
            print(f"[collect-prior] exact factorial: {nf} configurations (<= cap {max_configurations})", flush=True)
            assignment_iter = space.iter_factorial_assignments()
            total_plan = nf
            use_lhs = False
        else:
            print(
                f"[collect-prior] factorial {nf} > cap {max_configurations}; "
                f"switching to LHS ({max_configurations} samples, seed={lhs_seed})",
                flush=True,
            )
            assignment_iter = iter_lhs_unique_assignments(space, max_configurations, seed=lhs_seed)
            total_plan = max_configurations
            use_lhs = True

    knob_names: List[str] = space.knob_names
    baseline_assign: Optional[dict] = None
    if baseline is not None:
        baseline_assign = {k: baseline[k] for k in knob_names}
        print("[collect-prior] will evaluate prior-config baseline first (for PCR train row match)", flush=True)
    assignment_stream: Iterable = (
        itertools.chain((baseline_assign,), assignment_iter) if baseline_assign is not None else assignment_iter
    )
    loop_total = total_plan + (1 if baseline_assign is not None else 0)

    sizes = pd.read_csv(sizes_csv)
    if pgtuner_filename not in set(sizes["FileName"].astype(str)):
        raise ValueError(f"Add {pgtuner_filename} to {sizes_csv} columns FileName,N,q_queries")

    row = sizes[sizes["FileName"] == pgtuner_filename].iloc[0]
    n_base = float(row["N"])
    q_queries = float(row["q_queries"])

    knobs_meta = space.knobs_meta
    fieldnames = knob_names + [
        "construction_time",
        "memory",
        "recall",
        "search_time",
        "construct_dc_counts",
        "search_dc_counts",
    ]

    exist: set = set()
    if resume and out_csv.is_file():
        old = pd.read_csv(out_csv)
        if len(old):
            for _, r in old.iterrows():
                if str(r["FileName"]) != pgtuner_filename:
                    continue
                ad = {k: r[k] for k in knob_names}
                exist.add(_row_resume_key(pgtuner_filename, knob_names, knobs_meta, ad))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not resume and out_csv.is_file():
        out_csv.unlink()
    if not out_csv.is_file():
        with open(out_csv, "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["FileName"] + fieldnames)

    n_done = 0
    lhs_yielded = 0
    for assignment in assignment_stream:
        lhs_yielded += 1
        key = _row_resume_key(pgtuner_filename, knob_names, knobs_meta, assignment)
        if key in exist:
            continue
        full = space.build_full_milvus_config(assignment)
        work = dict(full)
        try:
            configure_index(*filter_index_rule(work))
            system_conf = {k: full[k] for k in full if "*" in k}
            configure_system(filter_system_rule(system_conf))
            prec, _rps, p95 = run_milvus_benchmark(
                engine, process_tag, benchmark_dataset, use_sudo=use_sudo, timeout_sec=timeout_sec
            )
        except Exception as e:
            print(
                f"[collect-prior] skip {assignment}: ({type(e).__name__}: {e})",
                flush=True,
            )
            traceback.print_exc()
            continue

        search_time = float(p95)
        recall = float(prec)
        construct_dc = _estimate_construct_dc_counts(full, n_base)
        search_dc = max(1, int(search_time * 1e6 * max(q_queries, 1.0)))

        out_row = [pgtuner_filename]
        for k in knob_names:
            v = assignment[k]
            if type(v) is bool:
                out_row.append(v)
            else:
                out_row.append(v)
        out_row.extend([0.0, 0, recall, search_time, construct_dc, search_dc])

        with open(out_csv, "a", encoding="utf-8", newline="") as fp:
            csv.writer(fp).writerow(out_row)
        exist.add(key)
        n_done += 1
        print(f"[collect-prior] {n_done}/{loop_total} logged recall={recall} p95={p95} {assignment}", flush=True)

    print(
        f"[collect-prior] finished (+{n_done} new rows); planned={loop_total}; factorial_ref={nf}; lhs={use_lhs}",
        flush=True,
    )
    if use_lhs and lhs_yielded < loop_total:
        print(
            f"[collect-prior] warning: stream ended after {lhs_yielded} assignment steps "
            f"(planned {loop_total}); discrete space may be finite or increase max_rounds in code",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Milvus-side QPP dataset tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Grid benchmark -> raw Milvus performance CSV")
    c.add_argument("--pgtuner-filename", required=True, help="Row key in data_feature.csv, e.g. tiny_1_1_384")
    c.add_argument("--benchmark-dataset", required=True)
    c.add_argument("--sizes-csv", type=Path, default=PGTUNER_ROOT / "Data/milvus_dataset_sizes.csv")
    c.add_argument("--out-csv", type=Path, default=PGTUNER_ROOT / "Data/index_performance_milvus.csv")
    c.add_argument("--grid", choices=("tiny", "medium", "large"), default="tiny")
    c.add_argument("--engine", default="milvus-single-node")
    c.add_argument("--process-tag", default="milvus-p10")
    c.add_argument("--no-sudo", action="store_true")
    c.add_argument("--timeout", type=int, default=2000)
    c.add_argument("--resume", action="store_true")

    m = sub.add_parser("merge", help="Merge Milvus perf + data_feature -> train_data.csv")
    m.add_argument("--data-feature", type=Path, default=PGTUNER_ROOT / "Data/vdb_data_feature.csv")
    m.add_argument("--milvus-perf", type=Path, required=True)
    m.add_argument("--sizes-csv", type=Path, default=PGTUNER_ROOT / "Data/milvus_dataset_sizes.csv")
    m.add_argument("--out", type=Path, default=PGTUNER_ROOT / "Data/train_data.csv")
    m.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="If set, keep only rows with this FileName (e.g. vdb_glove-100-angular)",
    )

    cp = sub.add_parser("collect-prior", help="Full factorial on prior tune_knobs -> Milvus perf CSV")
    cp.add_argument("--prior-config", type=Path, required=True)
    cp.add_argument("--knob-json", type=Path, default=None)
    cp.add_argument("--pgtuner-filename", required=True)
    cp.add_argument("--benchmark-dataset", required=True)
    cp.add_argument("--sizes-csv", type=Path, default=PGTUNER_ROOT / "Data/milvus_dataset_sizes.csv")
    cp.add_argument("--out-csv", type=Path, default=PGTUNER_ROOT / "Data/index_performance_milvus_prior.csv")
    cp.add_argument("--max-configurations", type=int, default=100_000)
    cp.add_argument(
        "--sampling",
        choices=("auto", "exact", "lhs"),
        default="auto",
        help="auto: factorial if fits else LHS; exact: full factorial only; lhs: always LHS with max-configurations points",
    )
    cp.add_argument("--lhs-seed", type=int, default=42, help="RNG seed for Latin Hypercube (each round uses seed+round)")
    cp.add_argument("--engine", default="milvus-single-node")
    cp.add_argument("--process-tag", default="milvus-p10")
    cp.add_argument("--no-sudo", action="store_true")
    cp.add_argument("--timeout", type=int, default=2000)
    cp.add_argument("--resume", action="store_true")

    args = p.parse_args()
    if args.cmd == "collect":
        collect_grid_rows_milvus(
            args.pgtuner_filename,
            args.sizes_csv,
            args.out_csv,
            args.grid,
            args.engine,
            args.process_tag,
            args.benchmark_dataset,
            use_sudo=not args.no_sudo,
            timeout_sec=args.timeout,
            resume=args.resume,
        )
    elif args.cmd == "merge":
        merge_milvus_perf_to_train_csv(
            args.data_feature, args.milvus_perf, args.sizes_csv, args.out, file_name=args.file_name
        )
    elif args.cmd == "collect-prior":
        vdir = Path(__file__).resolve().parent
        if str(vdir) not in sys.path:
            sys.path.insert(0, str(vdir))
        from vdtuner_prior_space import load_prior_bundle

        bundle = load_prior_bundle(args.knob_json, prior_config_path=args.prior_config)
        collect_prior_factorial_milvus(
            bundle.space,
            args.pgtuner_filename,
            args.sizes_csv,
            args.out_csv,
            args.benchmark_dataset,
            args.engine,
            args.process_tag,
            use_sudo=not args.no_sudo,
            timeout_sec=args.timeout,
            resume=args.resume,
            max_configurations=args.max_configurations,
            sampling=args.sampling,
            lhs_seed=args.lhs_seed,
            baseline=bundle.baseline,
        )
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
