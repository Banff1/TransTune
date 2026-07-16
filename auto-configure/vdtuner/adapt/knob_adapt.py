#!/usr/bin/env python3
"""
Data drift adaptation entrypoint.

Workflow:
1) Detect drift by RPS + mean_precision drop.
2) Strategy 1: tune search params on current index type.
3) Strategy 2: fix current index type, run Bayesian optimization on knobs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np
from numpy.lib.format import open_memmap

ADAPT_DIR = Path(__file__).resolve().parent
VDTUNER_DIR = ADAPT_DIR.parent
AUTO_CONFIGURE_ROOT = VDTUNER_DIR.parent
VDB_ROOT = AUTO_CONFIGURE_ROOT.parent
BENCHMARK_ROOT = VDB_ROOT / "vector-db-benchmark-master"
DRIFT_CYCLE_PY = ADAPT_DIR / "run_drift_cycle.py"
RESULTS_DIR = BENCHMARK_ROOT / "results"
CONF_PATH = BENCHMARK_ROOT / "experiments" / "configurations" / "milvus-single-node.json"
WHOLE_PARAM_PATH = AUTO_CONFIGURE_ROOT / "whole_param.json"
RUN_PY_PATH = BENCHMARK_ROOT / "run.py"
MILVUS_SERVER_PATH = "milvus-single-node"
RESET_MILVUS_SCRIPT = ADAPT_DIR / "reset_milvus_env.sh"
DEFAULT_DRIFT_STATE_FILE = ADAPT_DIR / ".drift_state.json"
DEFAULT_BACKUP_DIR = RESULTS_DIR / "drift_vector_backups"
DATASETS_JSON_PATH = BENCHMARK_ROOT / "datasets" / "datasets.json"
DATASETS_ROOT = BENCHMARK_ROOT / "datasets"
RESTORED_DATASET_ROOT = DATASETS_ROOT / "adapt-restored"

# Keep imports local-path based to match current project structure.
# `utils.py` imports `configure.py` from auto-configure root.
sys.path.insert(0, str(AUTO_CONFIGURE_ROOT))
sys.path.insert(0, str(VDTUNER_DIR))
sys.path.insert(0, str(BENCHMARK_ROOT))

PARTITION_BASED = {"IVF_FLAT", "IVF_SQ8", "IVF_PQ", "SCANN"}
GRAPH_BASED = {"HNSW"}


@dataclass
class BenchRunResult:
    result_json: dict[str, Any] | None
    result_path: Path | None
    error: str | None


def reset_milvus_service(
    clean_host_volumes: bool = True,
    max_wait_sec: int = 120,
    sudo_password: str = None  # 新增：如果需要，可以传入密码
) -> tuple[bool, str | None]:
    if not RESET_MILVUS_SCRIPT.exists():
        return False, f"reset script not found: {RESET_MILVUS_SCRIPT}"

    env = os.environ.copy()
    env["CLEAN_HOST_VOLUMES"] = "1" if clean_host_volumes else "0"
    
    # 核心改进：如果环境需要 sudo 且脚本内有 sudo 操作
    # 我们可以使用 ['sudo', '-S', str(RESET_MILVUS_SCRIPT), ...] 
    # 或者直接确保执行 Python 的用户拥有免密权限（见方案二）
    
    try:
        # 使用 input 参数将密码传给 sudo -S (如果脚本内部改写为支持 -S)
        cp = subprocess.run(
            [str(RESET_MILVUS_SCRIPT), MILVUS_SERVER_PATH, str(max_wait_sec)],
            cwd=str(BENCHMARK_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=max_wait_sec + 180,
            # 如果脚本里需要输密码，这里可以配合 sudo -S 传参
        )
    except Exception as exc:
        return False, f"执行异常: {exc}"

    if cp.returncode != 0:
        # 这里就是你看到的“清理失败”报错的地方
        msg = f"重置失败 (code={cp.returncode})\nError: {cp.stderr}"
        return False, msg

    return True, None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_drift_results(results_file: Path) -> list[dict[str, Any]]:
    if not results_file.exists():
        return []
    records: list[dict[str, Any]] = []
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_search_result(cycle: int, results_file: Path) -> dict[str, Any] | None:
    for record in load_drift_results(results_file):
        if record.get("cycle") != cycle:
            continue
        fname = record.get("file")
        if not fname:
            continue
        p = RESULTS_DIR / fname
        if p.exists():
            return _load_json(p)
    return None


def get_baseline_and_current(results_file: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    records = load_drift_results(results_file)
    if not records:
        return None, None, None
    baseline = _load_search_result(0, results_file)
    current_cycle = max(int(r["cycle"]) for r in records)
    current = _load_search_result(current_cycle, results_file)
    return baseline, current, current_cycle


def _read_metrics(payload: dict[str, Any]) -> tuple[float, float]:
    results = payload.get("results", {})
    rps = float(results.get("rps", 0.0) or 0.0)
    precision = float(results.get("mean_precisions", 0.0) or 0.0)
    return rps, precision


def detect_drift(
    baseline: dict[str, Any],
    current: dict[str, Any],
    rps_threshold: float,
    precision_threshold: float,
) -> bool:
    baseline_rps, baseline_precision = _read_metrics(baseline)
    current_rps, current_precision = _read_metrics(current)
    if baseline_rps <= 0 or baseline_precision <= 0:
        return False
    return (current_rps / baseline_rps) < rps_threshold or (current_precision / baseline_precision) < precision_threshold


def get_current_index_type() -> str:
    conf = _load_json(CONF_PATH)
    return conf[0].get("upload_params", {}).get("index_type", "SCANN")


def get_current_search_params() -> dict[str, Any]:
    conf = _load_json(CONF_PATH)
    return dict(conf[0].get("search_params", [{}])[0].get("params", {}))


def update_search_params(params: dict[str, Any]) -> None:
    conf = _load_json(CONF_PATH)
    conf[0]["search_params"][0]["params"] = params
    with CONF_PATH.open("w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)


def _latest_result_file(pattern: str) -> Path | None:
    files = [p for p in RESULTS_DIR.glob("*.json") if pattern in p.name and "search" in p.name]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def run_benchmark_full(dataset: str, engine: str, timeout_sec: int = 3600) -> tuple[bool, str | None]:
    """运行完整 benchmark（含 upload），用于重置后创建 collection 并导入初始数据。"""
    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    cmd = [
        sys.executable,
        str(RUN_PY_PATH),
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        "127.0.0.1",
        "--no-exit-on-error",
    ]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(BENCHMARK_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"benchmark timeout ({timeout_sec}s)"
    except Exception as exc:
        return False, f"benchmark exec error: {exc}"
    if cp.returncode != 0:
        return False, f"benchmark failed (code={cp.returncode})\n{(cp.stderr or '')[-800:]}"
    return True, None


def run_benchmark_skip_upload(dataset: str, engine: str, timeout_sec: int = 1800) -> BenchRunResult:
    pattern = f"{engine}-{dataset}-search"
    before_mtime = 0.0
    latest_before = _latest_result_file(pattern)
    if latest_before:
        before_mtime = latest_before.stat().st_mtime

    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    cmd = [
        sys.executable,
        str(RUN_PY_PATH),
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        "127.0.0.1",
        "--skip-upload",
        "--no-exit-on-error",
    ]

    try:
        cp = subprocess.run(
            cmd,
            cwd=str(BENCHMARK_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return BenchRunResult(None, None, f"benchmark timeout ({timeout_sec}s)")
    except Exception as exc:
        return BenchRunResult(None, None, f"benchmark exec error: {exc}")

    latest_after = _latest_result_file(pattern)
    if latest_after and latest_after.stat().st_mtime > before_mtime:
        try:
            return BenchRunResult(_load_json(latest_after), latest_after, None)
        except Exception as exc:
            return BenchRunResult(None, None, f"failed to parse result json: {exc}")

    stdout_tail = (cp.stdout or "")[-600:]
    stderr_tail = (cp.stderr or "")[-600:]
    return BenchRunResult(
        None,
        None,
        (
            f"benchmark failed (code={cp.returncode}).\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        ),
    )


def _init_drift_state_if_needed(dataset: str, state_file: Path, force: bool = False) -> tuple[bool, str | None]:
    if state_file.exists() and not force:
        return True, None
    cmd = [sys.executable, str(DRIFT_CYCLE_PY), "--get-initial-count", dataset]
    try:
        cp = subprocess.run(cmd, cwd=str(BENCHMARK_ROOT), capture_output=True, text=True, timeout=120)
        if cp.returncode != 0:
            return False, f"get-initial-count failed: {cp.stderr[-400:]}"
        initial_count = int((cp.stdout or "").strip())
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w", encoding="utf-8") as f:
            json.dump({"base_id": 0, "max_id": initial_count - 1}, f, indent=2)
        return True, None
    except Exception as exc:
        return False, f"init drift state failed: {exc}"


def _resolve_writable_state_file(preferred_state_file: Path) -> tuple[Path, str | None]:
    """
    Pick a writable state file path.
    If the preferred file exists but is not writable (e.g. root-owned), fallback to results dir.
    """
    preferred_state_file = preferred_state_file.expanduser().resolve()
    fallback_state_file = (RESULTS_DIR / ".drift_state.json").resolve()

    if preferred_state_file.exists():
        if os.access(preferred_state_file, os.W_OK):
            return preferred_state_file, None
        return (
            fallback_state_file,
            f"state file not writable: {preferred_state_file}; fallback to {fallback_state_file}",
        )

    parent = preferred_state_file.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return (
            fallback_state_file,
            f"cannot create parent dir for state file: {preferred_state_file}; fallback to {fallback_state_file}",
        )
    if os.access(parent, os.W_OK):
        return preferred_state_file, None
    return (
        fallback_state_file,
        f"state file parent not writable: {parent}; fallback to {fallback_state_file}",
    )


def simulate_drift_once(
    dataset: str,
    batch_size: int,
    state_file: Path,
    host: str = "127.0.0.1",
    port: int = 19530,
) -> tuple[bool, str | None, Path]:
    effective_state_file, warning = _resolve_writable_state_file(state_file)
    if warning:
        print(f"漂移状态文件告警: {warning}", flush=True)

    ok, err = _init_drift_state_if_needed(dataset=dataset, state_file=effective_state_file)
    if not ok:
        return False, err, effective_state_file
    cmd = [
        sys.executable,
        str(DRIFT_CYCLE_PY),
        "--dataset",
        dataset,
        "--batch-size",
        str(batch_size),
        "--host",
        host,
        "--port",
        str(port),
        "--state-file",
        str(effective_state_file),
    ]
    try:
        cp = subprocess.run(cmd, cwd=str(BENCHMARK_ROOT), capture_output=True, text=True, timeout=600)
        if cp.returncode != 0:
            return False, f"run_drift_cycle failed: {cp.stderr[-500:]}", effective_state_file
        print(cp.stdout.strip(), flush=True)
        return True, None, effective_state_file
    except Exception as exc:
        return False, f"simulate drift failed: {exc}", effective_state_file


def append_drift_result_record(results_file: Path, result_path: Path) -> int:
    records = load_drift_results(results_file)
    next_cycle = 0 if not records else (max(int(r["cycle"]) for r in records) + 1)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {"cycle": next_cycle, "file": result_path.name}
    with results_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return next_cycle


def backup_current_vectors(
    dataset: str,
    state_file: Path,
    backup_dir: Path,
    host: str = "127.0.0.1",
    port: int = 19530,
    chunk_size: int = 5000,
) -> tuple[bool, str]:
    try:
        from pymilvus import Collection, connections
        from engine.clients.milvus.config import MILVUS_COLLECTION_NAME, MILVUS_DEFAULT_ALIAS
    except Exception as exc:
        return False, f"pymilvus import failed: {exc}"

    if not state_file.exists():
        return False, f"state file not found: {state_file}"
    state = _load_json(state_file)
    base_id = int(state.get("base_id", 0))
    max_id = int(state.get("max_id", -1))
    if max_id < base_id:
        return False, f"invalid id range in state: base_id={base_id}, max_id={max_id}"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{dataset}-cycle-backup-{ts}.jsonl"
    meta_path = backup_dir / f"{dataset}-cycle-backup-{ts}.meta.json"

    connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=str(port))
    try:
        collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
        output_fields = [f.name for f in collection.schema.fields]
        total_rows = 0
        with backup_path.open("w", encoding="utf-8") as out:
            for start in range(base_id, max_id + 1, chunk_size):
                end = min(start + chunk_size, max_id + 1)
                expr = f"id >= {start} and id < {end}"
                rows = collection.query(expr=expr, output_fields=output_fields)
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += len(rows)
        meta = {
            "dataset": dataset,
            "created_at": ts,
            "state_file": str(state_file),
            "base_id": base_id,
            "max_id": max_id,
            "chunk_size": chunk_size,
            "row_count": total_rows,
            "backup_file": str(backup_path),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return True, str(backup_path)
    except Exception as exc:
        return False, f"backup failed: {exc}"
    finally:
        try:
            connections.disconnect(MILVUS_DEFAULT_ALIAS)
        except Exception:
            pass


def _load_dataset_configs() -> list[dict[str, Any]]:
    with DATASETS_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_dataset_configs(configs: list[dict[str, Any]]) -> None:
    with DATASETS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)


def _find_dataset_config(dataset_name: str) -> dict[str, Any] | None:
    for cfg in _load_dataset_configs():
        if cfg.get("name") == dataset_name:
            return cfg
    return None


def _metric_name(distance: str) -> str:
    d = (distance or "").lower()
    if d in ("cosine", "angular"):
        return "cosine"
    if d in ("l2", "euclidean"):
        return "l2"
    return "dot"


def _compute_knn_positions_for_queries(
    vectors: np.ndarray,
    query_rows: list[dict[str, Any]],
    distance: str,
    chunk_size: int = 5000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    metric = _metric_name(distance)
    n = vectors.shape[0]
    vector_norms = None
    if metric == "cosine":
        vector_norms = np.linalg.norm(vectors, axis=1) + 1e-12

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for row in query_rows:
        q = np.asarray(row["query"], dtype=np.float32)
        top_k = max(1, int(len(row.get("closest_ids", [])) or 10))
        top_k = min(top_k, n)

        if metric == "cosine":
            q_norm = float(np.linalg.norm(q) + 1e-12)

        best_scores = np.full(top_k, -np.inf, dtype=np.float32)
        best_pos = np.full(top_k, -1, dtype=np.int64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = vectors[start:end]
            if metric == "l2":
                diff = chunk - q
                scores = -np.einsum("ij,ij->i", diff, diff)
            elif metric == "cosine":
                scores = (chunk @ q) / (vector_norms[start:end] * q_norm)
            else:
                scores = chunk @ q

            if scores.shape[0] > top_k:
                local_idx = np.argpartition(scores, -top_k)[-top_k:]
            else:
                local_idx = np.arange(scores.shape[0], dtype=np.int64)
            cand_scores = scores[local_idx]
            cand_pos = (local_idx + start).astype(np.int64)

            merged_scores = np.concatenate([best_scores, cand_scores.astype(np.float32)])
            merged_pos = np.concatenate([best_pos, cand_pos])
            keep = np.argpartition(merged_scores, -top_k)[-top_k:]
            best_scores = merged_scores[keep]
            best_pos = merged_pos[keep]

        order = np.argsort(-best_scores)
        results.append((best_pos[order], best_scores[order]))

    return results


def _write_tests_with_ids(
    query_rows: list[dict[str, Any]],
    knn_pos_scores: list[tuple[np.ndarray, np.ndarray]],
    id_mapping: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row, (pos, scores) in zip(query_rows, knn_pos_scores):
            new_row = dict(row)
            new_row["closest_ids"] = [int(id_mapping[int(i)]) for i in pos.tolist()]
            new_row["closest_scores"] = [float(s) for s in scores.tolist()]
            f.write(json.dumps(new_row, ensure_ascii=False) + "\n")


def _prepare_eval_datasets_from_backup(
    source_dataset: str,
    backup_jsonl: Path,
) -> tuple[bool, str | None, str | None, str]:
    if not backup_jsonl.exists():
        return False, None, None, f"backup jsonl not found: {backup_jsonl}"

    src_cfg = _find_dataset_config(source_dataset)
    if not src_cfg:
        return False, None, None, f"source dataset config not found: {source_dataset}"
    src_path = DATASETS_ROOT / src_cfg["path"]
    tests_src = src_path / "tests.jsonl"
    if not tests_src.exists():
        return False, None, None, f"source tests.jsonl not found: {tests_src}"

    live_eval_name = f"{source_dataset}-drift-live-eval"
    reload_eval_name = f"{source_dataset}-drift-reload-eval"
    live_dir = RESTORED_DATASET_ROOT / live_eval_name
    reload_dir = RESTORED_DATASET_ROOT / reload_eval_name
    live_dir.mkdir(parents=True, exist_ok=True)
    reload_dir.mkdir(parents=True, exist_ok=True)

    vectors_npy = reload_dir / "vectors.npy"
    source_ids_npy = reload_dir / "source_ids.npy"
    payloads_jsonl = reload_dir / "payloads.jsonl"
    tests_live = live_dir / "tests.jsonl"
    tests_reload = reload_dir / "tests.jsonl"

    row_count = 0
    vector_dim = 0
    has_payload = False
    with backup_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vec = row.get("vector")
            if vec is None:
                return False, None, None, "backup row missing vector field"
            if vector_dim == 0:
                vector_dim = len(vec)
            row_count += 1
            if any(k not in ("id", "vector") for k in row.keys()):
                has_payload = True
    if row_count == 0 or vector_dim == 0:
        return False, None, None, "backup jsonl is empty"

    mmap = open_memmap(vectors_npy, mode="w+", dtype=np.float32, shape=(row_count, vector_dim))
    ids_mmap = open_memmap(source_ids_npy, mode="w+", dtype=np.int64, shape=(row_count,))
    payload_fp = payloads_jsonl.open("w", encoding="utf-8") if has_payload else None
    try:
        idx = 0
        with backup_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mmap[idx] = np.asarray(row["vector"], dtype=np.float32)
                ids_mmap[idx] = int(row.get("id", idx))
                if payload_fp is not None:
                    payload = {k: v for k, v in row.items() if k not in ("id", "vector")}
                    payload_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                idx += 1
    finally:
        del mmap
        del ids_mmap
        if payload_fp is not None:
            payload_fp.close()

    query_rows: list[dict[str, Any]] = []
    with tests_src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_rows.append(json.loads(line))
    if not query_rows:
        return False, None, None, f"tests.jsonl has no queries: {tests_src}"

    vectors = np.load(vectors_npy, mmap_mode="r")
    source_ids = np.load(source_ids_npy, mmap_mode="r")
    knn_pos_scores = _compute_knn_positions_for_queries(
        vectors=vectors,
        query_rows=query_rows,
        distance=str(src_cfg.get("distance", "cosine")),
    )
    _write_tests_with_ids(query_rows, knn_pos_scores, np.asarray(source_ids, dtype=np.int64), tests_live)
    _write_tests_with_ids(
        query_rows,
        knn_pos_scores,
        np.arange(row_count, dtype=np.int64),
        tests_reload,
    )

    cfgs = _load_dataset_configs()
    cfgs = [c for c in cfgs if c.get("name") not in {live_eval_name, reload_eval_name}]
    live_cfg: dict[str, Any] = {
        "name": live_eval_name,
        "vector_size": src_cfg["vector_size"],
        "distance": src_cfg["distance"],
        "type": "tar",
        "path": str(live_dir.relative_to(DATASETS_ROOT)),
    }
    reload_cfg: dict[str, Any] = {
        "name": reload_eval_name,
        "vector_size": src_cfg["vector_size"],
        "distance": src_cfg["distance"],
        "type": "tar",
        "path": str(reload_dir.relative_to(DATASETS_ROOT)),
    }
    if "schema" in src_cfg:
        live_cfg["schema"] = src_cfg["schema"]
        reload_cfg["schema"] = src_cfg["schema"]
    cfgs.append(live_cfg)
    cfgs.append(reload_cfg)
    _save_dataset_configs(cfgs)

    return (
        True,
        live_eval_name,
        reload_eval_name,
        f"recomputed KNN and prepared datasets: live={live_dir}, reload={reload_dir}, rows={row_count}, queries={len(query_rows)}",
    )


def _real_knob_conf(env: Any, x_vec: list[float]) -> dict[str, Any]:
    conf: dict[str, Any] = {}
    for idx, name in enumerate(env.names):
        _, real = env.knob_stand.scale_back(name, x_vec[idx])
        conf[name] = real
    return conf


def _print_config_diff(initial_conf: dict[str, Any], best_conf: dict[str, Any]) -> None:
    changed = [k for k in initial_conf.keys() if initial_conf.get(k) != best_conf.get(k)]
    print("  BO configuration comparison:", flush=True)
    print(f"    initial index_type: {initial_conf.get('index_type')}", flush=True)
    print(f"    final   index_type: {best_conf.get('index_type')}", flush=True)
    if not changed:
        print("    no knob changes", flush=True)
        return
    for k in changed:
        print(f"    {k}: {initial_conf.get(k)} -> {best_conf.get(k)}", flush=True)


def _clamp_int(v: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(v)))


def _bounds_for(name: str, fallback_min: int, fallback_max: int) -> tuple[int, int]:
    if WHOLE_PARAM_PATH.exists():
        cfg = _load_json(WHOLE_PARAM_PATH)
        if name in cfg and isinstance(cfg[name], dict):
            return int(cfg[name].get("min", fallback_min)), int(cfg[name].get("max", fallback_max))
    return fallback_min, fallback_max


def strategy1_param_adjustment(
    index_type: str,
    current_params: dict[str, Any],
    baseline: dict[str, Any],
    dataset: str,
    engine: str,
    max_tries: int,
) -> bool:
    baseline_rps, baseline_precision = _read_metrics(baseline)
    target_rps = baseline_rps * 0.95
    target_precision = baseline_precision * 0.95
    original_params = dict(current_params)

    if index_type in PARTITION_BASED:
        nprobe_min, nprobe_max = _bounds_for("nprobe", 1, 100)
        reorder_min, reorder_max = _bounds_for("reorder_k", 1, 2000)
        nprobe_base = _clamp_int(int(current_params.get("nprobe", nprobe_min)), nprobe_min, nprobe_max)
        reorder_base = _clamp_int(int(current_params.get("reorder_k", reorder_min)), reorder_min, reorder_max)

        for i in range(max_tries):
            factor = 1.0 + (i + 1) * 0.15
            candidate = {
                "nprobe": _clamp_int(round(nprobe_base * factor), nprobe_min, nprobe_max),
                "reorder_k": _clamp_int(round(reorder_base * (1.0 + (i + 1) * 0.10)), reorder_min, reorder_max),
            }
            update_search_params(candidate)
            print(f"  Strategy 1 try {i + 1}: {candidate}", flush=True)
            bench = run_benchmark_skip_upload(dataset=dataset, engine=engine)
            if bench.result_json is None:
                print(f"  Strategy 1 benchmark failed: {bench.error}", flush=True)
                continue
            rps, precision = _read_metrics(bench.result_json)
            print(f"  -> RPS={rps:.1f}, mean_precision={precision:.4f}", flush=True)
            if rps >= target_rps and precision >= target_precision:
                print("  Strategy 1 成功恢复", flush=True)
                return True

    elif index_type in GRAPH_BASED:
        ef_min, ef_max = _bounds_for("ef", 1, 2000)
        ef_base = _clamp_int(int(current_params.get("ef", ef_min)), ef_min, ef_max)
        for i in range(max_tries):
            candidate = {"ef": _clamp_int(round(ef_base * (1.0 + (i + 1) * 0.15)), ef_min, ef_max)}
            update_search_params(candidate)
            print(f"  Strategy 1 try {i + 1}: {candidate}", flush=True)
            bench = run_benchmark_skip_upload(dataset=dataset, engine=engine)
            if bench.result_json is None:
                print(f"  Strategy 1 benchmark failed: {bench.error}", flush=True)
                continue
            rps, precision = _read_metrics(bench.result_json)
            print(f"  -> RPS={rps:.1f}, mean_precision={precision:.4f}", flush=True)
            if rps >= target_rps and precision >= target_precision:
                print("  Strategy 1 成功恢复", flush=True)
                return True
    else:
        print(f"  Strategy 1: index_type={index_type} 无可调搜索参数，跳过", flush=True)
        return False

    # Strategy 1 did not recover; roll back config to original search params.
    update_search_params(original_params)
    print(f"  Strategy 1 失败，已回滚搜索参数到: {original_params}", flush=True)
    return False


def _check_bo_success(model: Any, index_type: str, baseline: dict[str, Any]) -> tuple[bool, float, float]:
    target_rps, target_precision = _read_metrics(baseline)
    target_rps *= 0.95
    target_precision *= 0.95

    best_rps = 0.0
    best_precision = 0.0
    for item in model.Y.get(index_type, []):
        if not item:
            continue
        precision = float(item[0])
        rps = float(item[1])
        best_rps = max(best_rps, rps)
        best_precision = max(best_precision, precision)
        if rps >= target_rps and precision >= target_precision:
            return True, best_rps, best_precision
    return False, best_rps, best_precision


def strategy2_bayesian_optimization(
    index_type: str,
    baseline: dict[str, Any],
    dataset_for_tuning: str,
    max_rounds: int,
) -> bool:
    try:
        from adapt.optimizer_pobo_sa_adapt import PollingBayesianOptimization
        from utils import RealEnv
    except Exception as exc:
        print("Strategy 2 依赖导入失败，请确认 venv 中依赖可用。", flush=True)
        print(f"导入错误: {exc}", flush=True)
        return False

    try:
        print(f"Strategy 2 使用恢复数据集: {dataset_for_tuning}", flush=True)
        
        # 定义内部函数：在 RealEnv 交互前重置环境
        def env_with_reset_wrapper(dataset_name):
            # 每次评估前清理 Milvus 容器和 Volume
            # 注意：此处 clean_host_volumes=True 确保彻底删除旧索引文件
            success, err = reset_milvus_service(clean_host_volumes=True)
            if not success:
                print(f"警告: Strategy 2 迭代中重置 Milvus 失败: {err}")
            return RealEnv(dataset=dataset_name)

        env = env_with_reset_wrapper(dataset_for_tuning)
        tune_knobs = [k for k in env.names if k != "index_type"]
        model = PollingBayesianOptimization(
            env,
            seed=42,
            allowed_index_types=[index_type],
            tune_knobs=tune_knobs,
        )

        model.init_sample()
        success, best_rps, best_precision = _check_bo_success(model, index_type, baseline)
        print(f"  BO init: best_rps={best_rps:.1f}, best_precision={best_precision:.4f}", flush=True)
        best_idx = 0
        best_score = -1.0
        for i, row in enumerate(model.Y.get(index_type, [])):
            if not row:
                continue
            score = float(row[0]) * float(row[1])
            if score > best_score:
                best_score = score
                best_idx = i
        if success:
            initial_conf = _real_knob_conf(env, model.X[index_type][0])
            best_conf = _real_knob_conf(env, model.X[index_type][best_idx])
            _print_config_diff(initial_conf, best_conf)
            return True

        for step in range(max_rounds):
            model.step()
            success, best_rps, best_precision = _check_bo_success(model, index_type, baseline)
            for i, row in enumerate(model.Y.get(index_type, [])):
                if not row:
                    continue
                score = float(row[0]) * float(row[1])
                if score > best_score:
                    best_score = score
                    best_idx = i
            print(
                f"  BO step {step + 1}/{max_rounds}: best_rps={best_rps:.1f}, best_precision={best_precision:.4f}",
                flush=True,
            )
            if success:
                initial_conf = _real_knob_conf(env, model.X[index_type][0])
                best_conf = _real_knob_conf(env, model.X[index_type][best_idx])
                _print_config_diff(initial_conf, best_conf)
                return True
        if model.X.get(index_type):
            initial_conf = _real_knob_conf(env, model.X[index_type][0])
            best_conf = _real_knob_conf(env, model.X[index_type][best_idx])
            _print_config_diff(initial_conf, best_conf)
        return False
    except Exception:
        print("Strategy 2 运行异常：", flush=True)
        print(traceback.format_exc(), flush=True)
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data drift adaptation (search params + BO).")
    parser.add_argument("--results-file", default=None, help="Path to drift_test_results.jsonl")
    parser.add_argument("--rps-threshold", type=float, default=0.9)
    parser.add_argument("--precision-threshold", type=float, default=0.9)
    parser.add_argument("--dataset", default="random-geo-radius-2048-angular-no-filters")
    parser.add_argument("--engine", default="milvus-p10")
    parser.add_argument("--strategy1-tries", type=int, default=5)
    parser.add_argument("--strategy2-rounds", type=int, default=30)
    parser.add_argument("--skip-strategy", choices=["1", "2", "none"], default="none")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--simulate-drift-before-adapt", action="store_true", default=True)
    parser.add_argument("--no-simulate-drift-before-adapt", action="store_false", dest="simulate_drift_before_adapt")
    parser.add_argument("--drift-batch-size", type=int, default=3000)
    parser.add_argument("--drift-state-file", default=str(DEFAULT_DRIFT_STATE_FILE))
    parser.add_argument("--drift-host", default="127.0.0.1")
    parser.add_argument("--drift-port", type=int, default=19530)
    parser.add_argument("--backup-vectors-before-adapt", action="store_true", default=True)
    parser.add_argument("--no-backup-vectors-before-adapt", action="store_false", dest="backup_vectors_before_adapt")
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--reset-milvus-first", action="store_true", default=True)
    parser.add_argument("--no-reset-milvus-first", action="store_false", dest="reset_milvus_first")
    parser.add_argument("--no-clean-host-volumes", action="store_true", default=False)
    parser.add_argument("--milvus-max-wait-sec", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results_file = Path(args.results_file) if args.results_file else (RESULTS_DIR / "drift_test_results.jsonl")

    print("=" * 50)
    print("Knob Adapt: 数据漂移检测与自适应")
    print("=" * 50)

    # --- 新增清理逻辑 ---
    backup_dir = Path(args.backup_dir)
    if backup_dir.exists():
        print(f">>> 清理旧的备份目录: {backup_dir}")
        try:
            # 清理目录下的所有文件，但保留目录本身
            for item in backup_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        except Exception as e:
            print(f"警告: 清理备份目录时出错: {e}")
    # ------------------    

    if args.reset_milvus_first:
        ok, err = reset_milvus_service(
            clean_host_volumes=not args.no_clean_host_volumes,
            max_wait_sec=args.milvus_max_wait_sec,
        )
        if not ok:
            print(f"Milvus 重置失败: {err}", flush=True)
            return 1

    baseline, current_from_file, _current_cycle = get_baseline_and_current(results_file)
    if baseline is None:
        print("错误: 无法加载 drift 结果，请先运行 auto-configure/vdtuner/adapt/run_drift_test.sh")
        return 1

    current = current_from_file
    current_cycle: int | str | None = _current_cycle

    effective_state_file, state_warn = _resolve_writable_state_file(Path(args.drift_state_file))
    if state_warn:
        print(f"漂移状态文件: {state_warn}", flush=True)
    if args.simulate_drift_before_adapt:
        # 重置后 Milvus 为空，需先完整导入以创建 collection，否则 run_drift_cycle 会报 SchemaNotReadyException
        print(">>> 预处理: 初始导入数据（创建 collection）...", flush=True)
        ok, err = run_benchmark_full(dataset=args.dataset, engine=args.engine)
        if not ok:
            print(f"初始导入失败: {err}", flush=True)
            return 1
        print("    初始导入完成", flush=True)
        # 初始化漂移状态（与 run_drift_test.sh 一致，强制覆盖以匹配刚导入的数据）
        init_ok, init_err = _init_drift_state_if_needed(
            dataset=args.dataset, state_file=effective_state_file, force=True
        )
        if not init_ok:
            print(f"漂移状态初始化失败: {init_err}", flush=True)
            return 1
        print(">>> 预处理: 执行一轮 run_drift_cycle.py 以模拟最新数据漂移", flush=True)
        ok, err, effective_state_file = simulate_drift_once(
            dataset=args.dataset,
            batch_size=args.drift_batch_size,
            state_file=effective_state_file,
            host=args.drift_host,
            port=args.drift_port,
        )
        if not ok:
            print(f"漂移模拟失败: {err}", flush=True)
            return 1

        bench = run_benchmark_skip_upload(dataset=args.dataset, engine=args.engine)
        if bench.result_json is None:
            print(f"漂移后 benchmark 失败: {bench.error}", flush=True)
            return 1
        current = bench.result_json
        if bench.result_path is not None:
            current_cycle = append_drift_result_record(results_file=results_file, result_path=bench.result_path)
        else:
            current_cycle = "simulated"
    elif current is None:
        print("错误: 当前 cycle 结果不存在，且未启用漂移模拟")
        return 1

    baseline_rps, baseline_precision = _read_metrics(baseline)
    current_rps, current_precision = _read_metrics(current)
    print(f"Baseline (cycle 0): RPS={baseline_rps:.1f}, mean_precision={baseline_precision:.4f}")
    print(f"Current (cycle {current_cycle}): RPS={current_rps:.1f}, mean_precision={current_precision:.4f}")

    drifted = detect_drift(
        baseline=baseline,
        current=current,
        rps_threshold=args.rps_threshold,
        precision_threshold=args.precision_threshold,
    )
    if not drifted:
        print("未检测到 data drifting，无需 adapt。")
        return 0

    if args.detect_only:
        print("检测到 data drifting（--detect-only，不执行 adapt）")
        return 0

    print("检测到 data drifting，开始 Knob Adapt...")

    backup_jsonl_path: Path | None = None
    strategy1_dataset = args.dataset
    strategy2_dataset: str | None = None
    if args.backup_vectors_before_adapt:
        ok, msg = backup_current_vectors(
            dataset=args.dataset,
            state_file=effective_state_file,
            backup_dir=Path(args.backup_dir),
            host=args.drift_host,
            port=args.drift_port,
        )
        if ok:
            print(f"当前向量数据已备份: {msg}", flush=True)
            backup_jsonl_path = Path(msg)
            prep_ok, live_ds, reload_ds, prep_msg = _prepare_eval_datasets_from_backup(
                source_dataset=args.dataset,
                backup_jsonl=backup_jsonl_path,
            )
            if prep_ok and live_ds and reload_ds:
                strategy1_dataset = live_ds
                strategy2_dataset = reload_ds
                print(f"已重算 query KNN: {prep_msg}", flush=True)
                print(f"  Strategy 1 dataset: {strategy1_dataset}", flush=True)
                print(f"  Strategy 2 dataset: {strategy2_dataset}", flush=True)
            else:
                print(f"KNN 重算失败（使用原 dataset）: {prep_msg}", flush=True)
        else:
            print(f"向量备份失败（继续执行 adapt）: {msg}", flush=True)

    index_type = get_current_index_type()
    current_params = get_current_search_params()
    print(f"当前索引类型: {index_type}, 搜索参数: {current_params}")

    if args.skip_strategy != "1":
        print("\n>>> Strategy 1: 调整搜索参数")
        if strategy1_param_adjustment(
            index_type=index_type,
            current_params=current_params,
            baseline=baseline,
            dataset=strategy1_dataset,
            engine=args.engine,
            max_tries=args.strategy1_tries,
        ):
            print("Knob Adapt 成功 (Strategy 1)")
            return 0

    if args.skip_strategy != "2":
        print("\n>>> Strategy 2: 贝叶斯优化 (固定索引类型)")
        if strategy2_dataset is None:
            print("Strategy 2 跳过：未准备好恢复数据集（请开启备份并确保 KNN 重算成功）。", flush=True)
        elif strategy2_bayesian_optimization(
            index_type=index_type,
            baseline=baseline,
            dataset_for_tuning=strategy2_dataset,
            max_rounds=args.strategy2_rounds,
        ):
            print("Knob Adapt 成功 (Strategy 2)")
            return 0

    print("\n" + "=" * 50)
    print("Knob Adapt 失败：需要完全重新进行 tuning")
    print("建议：重新运行 full tuning (main_tuner.py)，考虑更换索引类型或重新采集数据")
    print("=" * 50)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

