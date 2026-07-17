#!/usr/bin/env python3
"""
Export Milvus corpus (after drift) as vectors.npy + source_ids.npy + tests.jsonl with
exact KNN labels relative to that corpus.

Used by run_drift_cycle.py after each drift (and optionally for baseline via --export-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

ADAPT_DIR = Path(__file__).resolve().parent
VDTUNER_DIR = ADAPT_DIR.parent
AUTO_CONFIGURE_ROOT = VDTUNER_DIR.parent
VDB_ROOT = AUTO_CONFIGURE_ROOT.parent
BENCHMARK_ROOT = VDB_ROOT / "vector-db-benchmark-master"
DATASETS_JSON_PATH = BENCHMARK_ROOT / "datasets" / "datasets.json"
DATASETS_ROOT = BENCHMARK_ROOT / "datasets"


def _ensure_benchmark_path() -> None:
    p = str(BENCHMARK_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_dataset_config(dataset_name: str) -> dict[str, Any]:
    with DATASETS_JSON_PATH.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    for cfg in configs:
        if cfg.get("name") == dataset_name:
            return cfg
    raise ValueError(f"Dataset '{dataset_name}' not found in datasets.json")


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row, (pos, scores) in zip(query_rows, knn_pos_scores):
            new_row = dict(row)
            new_row["closest_ids"] = [int(id_mapping[int(i)]) for i in pos.tolist()]
            new_row["closest_scores"] = [float(s) for s in scores.tolist()]
            f.write(json.dumps(new_row, ensure_ascii=False) + "\n")


def _sorted_milvus_rows(
    *,
    dataset_name: str,
    base_id: int,
    max_id: int,
    host: str,
    port: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    _ensure_benchmark_path()
    from pymilvus import Collection, connections
    from engine.clients.milvus.config import MILVUS_COLLECTION_NAME, MILVUS_DEFAULT_ALIAS

    connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=str(port))
    try:
        collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
        output_fields = [f.name for f in collection.schema.fields]
        batches: list[dict[str, Any]] = []
        for start in range(base_id, max_id + 1, chunk_size):
            end = min(start + chunk_size, max_id + 1)
            expr = f"id >= {start} and id < {end}"
            rows = collection.query(expr=expr, output_fields=output_fields)
            batches.extend(rows)
        batches.sort(key=lambda r: int(r["id"]))
        if len(batches) != (max_id - base_id + 1):
            got = len(batches)
            exp = max_id - base_id + 1
            raise RuntimeError(
                f"Milvus row count mismatch after drift: expected {exp} vectors in [{base_id}, {max_id}], got {got}"
            )
        return batches
    finally:
        try:
            connections.disconnect(MILVUS_DEFAULT_ALIAS)
        except Exception:
            pass


def _vector_from_row(row: dict[str, Any]) -> list[float]:
    v = row.get("vector")
    if v is not None:
        return v
    for _k, val in row.items():
        if _k != "id" and isinstance(val, list) and val and isinstance(val[0], (int, float)):
            return val
    raise ValueError("cannot infer vector column from Milvus row")


def _row_vectors_and_ids(sorted_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not sorted_rows:
        raise ValueError("no rows from Milvus")
    vecs = np.stack([np.asarray(_vector_from_row(r), dtype=np.float32) for r in sorted_rows], axis=0)
    ids = np.array([int(r["id"]) for r in sorted_rows], dtype=np.int64)
    return vecs, ids


def export_post_drift_snapshot(
    *,
    dataset_name: str,
    state_file: Path,
    export_dir: Path,
    host: str = "127.0.0.1",
    port: int = 19530,
    chunk_size: int = 5000,
    cycle_note: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Writes under export_dir:
      - vectors.npy, source_ids.npy (row order follows ascending Milvus id)
      - tests.jsonl (queries from original dataset tests.jsonl, closest_* = exact KNN in **live Milvus id** space)
      - tests_reload_eval.jsonl (same KNN mapped to contiguous row ids 0..N-1 for reload benchmarks)
      - snapshot_meta.json (metadata)
    """
    try:
        if not state_file.exists():
            return False, f"state file not found: {state_file}"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        base_id = int(state["base_id"])
        max_id = int(state["max_id"])
        if max_id < base_id:
            return False, f"invalid id range in state: base_id={base_id}, max_id={max_id}"

        cfg = load_dataset_config(dataset_name)
        src_path = DATASETS_ROOT / cfg["path"]
        tests_src = src_path / "tests.jsonl"
        if not tests_src.exists():
            return False, f"source tests.jsonl not found: {tests_src}"

        rows = _sorted_milvus_rows(
            dataset_name=dataset_name,
            base_id=base_id,
            max_id=max_id,
            host=host,
            port=port,
            chunk_size=chunk_size,
        )
        vectors, source_ids_arr = _row_vectors_and_ids(rows)
        export_dir.mkdir(parents=True, exist_ok=True)

        mmap = open_memmap(export_dir / "vectors.npy", mode="w+", dtype=np.float32, shape=vectors.shape)
        mmap[:] = vectors[:]
        del mmap

        ids_mmap = open_memmap(export_dir / "source_ids.npy", mode="w+", dtype=np.int64, shape=source_ids_arr.shape)
        ids_mmap[:] = source_ids_arr[:]
        del ids_mmap

        query_rows: list[dict[str, Any]] = []
        with tests_src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    query_rows.append(json.loads(line))
        if not query_rows:
            return False, f"tests.jsonl has no queries: {tests_src}"

        knn = _compute_knn_positions_for_queries(
            vectors=np.load(export_dir / "vectors.npy", mmap_mode="r"),
            query_rows=query_rows,
            distance=str(cfg.get("distance", "cosine")),
        )

        reload_ids = np.arange(vectors.shape[0], dtype=np.int64)
        _write_tests_with_ids(query_rows, knn, source_ids_arr, export_dir / "tests.jsonl")
        _write_tests_with_ids(query_rows, knn, reload_ids, export_dir / "tests_reload_eval.jsonl")

        meta: dict[str, Any] = {
            "dataset": dataset_name,
            "base_id": base_id,
            "max_id": max_id,
            "num_vectors": int(vectors.shape[0]),
            "vector_dim": int(vectors.shape[1]),
            "distance": cfg.get("distance"),
            "source_tests": str(tests_src),
            "state_file": str(state_file),
            "files": {
                "vectors.npy": "corpus vectors (row-major, sorted by ascending Milvus id)",
                "source_ids.npy": "Milvus id per row index",
                "tests.jsonl": "queries with exact KNN in live Milvus id space",
                "tests_reload_eval.jsonl": "same KNN with neighbor ids relabeled as 0..N-1",
            },
        }
        if cycle_note is not None:
            meta["cycle_note"] = cycle_note
        if extra_meta:
            meta.update(extra_meta)
        (export_dir / "snapshot_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return True, str(export_dir.resolve())
    except Exception as exc:
        return False, f"snapshot export failed: {exc}"
