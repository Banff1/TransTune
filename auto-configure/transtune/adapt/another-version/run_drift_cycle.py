#!/usr/bin/env python3
"""
数据漂移：向 Milvus 插入一批新向量并删除最旧的一批（插入条数与删除条数可不等）。

漂移插入向量由 generate_drift_insert_vectors 按 --drift-insert-dist 采样，默认 laplace，
与基准数据集中常见的独立高斯各维再归一化的几何/statistics 刻意拉开差距。

KNN / 导出（均在本文件内实现，不依赖其他目录脚本）：
  --write-live-tests   从 Milvus 读当前全集到内存，按基准数据集的查询向量重算精确 KNN，
                       只写一个 tests.jsonl（供 skip-upload 搜索阶段算 precision）。
  --export-final       仅在收尾时写出 vectors.npy + tests.jsonl 各一份（与当前 Milvus 一致）。

查询与原始 ground truth 来源：
  type=h5  ：从 .hdf5 的 test / neighbors / distances 构造（无 tests.jsonl）
  type=tar ：读取数据集目录下 tests.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DRIFTING_DIR = Path(__file__).resolve().parent
VDB_TUNING_ROOT = DRIFTING_DIR.parent.parent.parent.parent
BENCHMARK_ROOT = VDB_TUNING_ROOT / "vector-db-benchmark-master"
DATASETS_ROOT = BENCHMARK_ROOT / "datasets"

sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
from numpy.lib.format import open_memmap
from pymilvus import Collection, connections, utility

from engine.clients.milvus.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)


def load_dataset_config(dataset_name: str) -> dict[str, Any]:
    datasets_path = BENCHMARK_ROOT / "datasets" / "datasets.json"
    with open(datasets_path, "r") as f:
        configs = json.load(f)
    for cfg in configs:
        if cfg["name"] == dataset_name:
            return cfg
    raise ValueError(f"Dataset '{dataset_name}' not found in datasets.json")


def get_initial_count(dataset_name: str) -> int:
    cfg = load_dataset_config(dataset_name)
    base = DATASETS_ROOT
    path_str = cfg["path"]
    if cfg["type"] in ("tar",) or any(
        x in dataset_name
        for x in (
            "random-geo",
            "random-match",
            "random-range",
            "h-and-m",
            "arxiv",
            "yandex",
            "dbpedia",
            "laion",
        )
    ):
        vectors_file = base / path_str / "vectors.npy"
        if vectors_file.exists():
            return int(np.load(vectors_file).shape[0])
    elif cfg["type"] == "h5":
        import h5py

        with h5py.File(base / path_str, "r") as f:
            return int(f["train"].shape[0])
    elif cfg["type"] == "jsonl":
        vectors_file = base / path_str / "vectors.jsonl"
        if vectors_file.exists():
            with open(vectors_file) as f:
                return sum(1 for _ in f)
    raise ValueError(f"Cannot infer count for dataset {dataset_name}")


def load_query_rows_from_benchmark(dataset_name: str) -> list[dict[str, Any]]:
    """
    构造与 AnnCompoundReader.tests.jsonl 行兼容的 dict 列表（query / closest_ids / closest_scores / conditions）。
    - h5：自 HDF5 中读取 test、neighbors、distances（Milvus 漂移阶段会把 closest_* 重算为当前语料上的精确 KNN）。
    - tar：读取 path/tests.jsonl。
    """
    cfg = load_dataset_config(dataset_name)
    normalize = cfg.get("distance", "").lower() == "cosine"
    dtype = cfg["type"]

    if dtype == "h5":
        import h5py

        h5_path = DATASETS_ROOT / cfg["path"]
        if not h5_path.is_file():
            raise FileNotFoundError(f"hdf5 dataset file not found: {h5_path}")
        rows: list[dict[str, Any]] = []
        with h5py.File(h5_path, "r") as data:
            n_test = int(data["test"].shape[0])
            for i in range(n_test):
                v = np.asarray(data["test"][i], dtype=np.float64)
                if normalize:
                    nrm = float(np.linalg.norm(v))
                    if nrm > 1e-30:
                        v = v / nrm
                nb = np.asarray(data["neighbors"][i]).reshape(-1)
                ds = np.asarray(data["distances"][i]).reshape(-1)
                rows.append(
                    {
                        "query": v.astype(np.float32).tolist(),
                        "closest_ids": [int(x) for x in nb.tolist()],
                        "closest_scores": [float(x) for x in ds.tolist()],
                        "conditions": None,
                    }
                )
        if not rows:
            raise ValueError(f"no test vectors in {h5_path}")
        return rows

    if dtype == "tar":
        tests_path = DATASETS_ROOT / cfg["path"] / "tests.jsonl"
        if not tests_path.exists():
            raise FileNotFoundError(f"benchmark tests.jsonl not found: {tests_path}")
        rows = []
        with tests_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"no queries in {tests_path}")
        return rows

    raise ValueError(
        f"dataset type '{dtype}' is not supported for drift live KNN "
        f"(supported: h5, tar). Dataset: {dataset_name}"
    )


def generate_drift_insert_vectors(
    n: int,
    dim: int,
    normalize: bool,
    distribution: str,
) -> np.ndarray:
    """
    漂移插入专用向量，默认 intentionally 与「独立高斯各维再归一」的常见基准生成方式拉开差距。
    distribution:
      gaussian          — 各维 i.i.d. N(0,1)，与旧行为一致
      uniform_cube      — U[-1,1]^d，角分布与 Gaussian 归一后不同（尤其中低维）
      laplace           — 各维 Laplace(0,1)，重尾、邻域结构与 Gaussian 差异大
      sparse_orthogonal — 每向量仅少量随机坐标非零再归一化，几何上与稠密随机点云很不一致
    """
    dkey = distribution.strip().lower()
    if dkey == "gaussian":
        vectors = np.random.randn(n, dim).astype(np.float32)
    elif dkey == "uniform_cube":
        vectors = np.random.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
    elif dkey == "laplace":
        vectors = np.random.laplace(0.0, 1.0, size=(n, dim)).astype(np.float32)
    elif dkey == "sparse_orthogonal":
        k = max(4, min(dim, dim // 32 + 8))
        vectors = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            idx = np.random.choice(dim, size=k, replace=False)
            vectors[i, idx] = np.random.randn(k).astype(np.float32)
    else:
        raise ValueError(
            f"unknown drift insert distribution: {distribution!r} "
            "(use gaussian, uniform_cube, laplace, sparse_orthogonal)"
        )

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def run_drift_cycle(
    host: str = "127.0.0.1",
    port: int = 19530,
    dataset_name: str = "random-geo-radius-2048-angular-no-filters",
    insert_batch_size: int = 1000,
    delete_batch_size: int = 1000,
    drift_insert_distribution: str = "laplace",
    base_id: int = 0,
    max_id: int = 99999,
) -> tuple[int, int]:
    cfg = load_dataset_config(dataset_name)
    dim = cfg["vector_size"]
    normalize = cfg.get("distance", "").lower() == "cosine"

    connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=str(port))
    collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

    new_ids = list(range(max_id + 1, max_id + 1 + insert_batch_size))
    vectors = generate_drift_insert_vectors(
        insert_batch_size,
        dim,
        normalize,
        drift_insert_distribution,
    )
    collection.insert([new_ids, vectors.tolist()])

    delete_ids = list(range(base_id, base_id + delete_batch_size))
    ids_str = ",".join(str(i) for i in delete_ids)
    collection.delete(f"id in [{ids_str}]")

    collection.flush()
    connections.disconnect(MILVUS_DEFAULT_ALIAS)

    return base_id + delete_batch_size, max_id + insert_batch_size


def get_segment_vector_counts(
    host: str = "127.0.0.1",
    port: int = 19530,
) -> tuple[list[dict], int]:
    """返回 (segment 列表, collection.num_entities)。各 segment 的 num_rows 之和常不等于逻辑行数。"""
    connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=str(port))
    try:
        collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
        num_entities = int(collection.num_entities)

        seg_infos = utility.get_query_segment_info(
            collection_name=MILVUS_COLLECTION_NAME,
            using=MILVUS_DEFAULT_ALIAS,
        )
        result = []
        for seg in seg_infos:
            seg_id = getattr(seg, "segmentID", None) or getattr(seg, "segment_id", "?")
            num_rows = getattr(seg, "num_rows", None) or getattr(seg, "numRows", 0)
            state = getattr(seg, "state", "?")
            part_id = getattr(seg, "partitionID", None) or getattr(seg, "partition_id", "?")
            result.append(
                {
                    "segment_id": str(seg_id),
                    "num_rows": int(num_rows) if num_rows is not None else 0,
                    "state": str(state),
                    "partition_id": str(part_id),
                }
            )
        return result, num_entities
    finally:
        connections.disconnect(MILVUS_DEFAULT_ALIAS)


def _metric_name(distance: str) -> str:
    d = (distance or "").lower()
    if d in ("cosine", "angular"):
        return "cosine"
    if d in ("l2", "euclidean"):
        return "l2"
    return "dot"


def _sorted_milvus_rows(
    *,
    base_id: int,
    max_id: int,
    host: str,
    port: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=str(port))
    try:
        collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
        output_fields = [f.name for f in collection.schema.fields]
        batches: list[dict[str, Any]] = []
        for start in range(base_id, max_id + 1, chunk_size):
            end = min(start + chunk_size, max_id + 1)
            expr = f"id >= {start} and id < {end}"
            batches.extend(collection.query(expr=expr, output_fields=output_fields))
        batches.sort(key=lambda r: int(r["id"]))
        exp = max_id - base_id + 1
        if len(batches) != exp:
            raise RuntimeError(
                f"Milvus row count mismatch: expected {exp} vectors in [{base_id}, {max_id}], got {len(batches)}"
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


def _corpus_arrays(sorted_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vecs = np.stack([np.asarray(_vector_from_row(r), dtype=np.float32) for r in sorted_rows], axis=0)
    ids = np.array([int(r["id"]) for r in sorted_rows], dtype=np.int64)
    return vecs, ids


def _compute_knn_positions_for_queries(
    vectors: np.ndarray,
    query_rows: list[dict[str, Any]],
    distance: str,
    chunk_size: int = 5000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    metric = _metric_name(distance)
    n = vectors.shape[0]
    vector_norms = np.linalg.norm(vectors, axis=1) + 1e-12 if metric == "cosine" else None

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for row in query_rows:
        q = np.asarray(row["query"], dtype=np.float32)
        top_k = max(1, int(len(row.get("closest_ids", [])) or 10))
        top_k = min(top_k, n)
        q_norm = float(np.linalg.norm(q) + 1e-12) if metric == "cosine" else 0.0

        best_scores = np.full(top_k, -np.inf, dtype=np.float32)
        best_pos = np.full(top_k, -1, dtype=np.int64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = vectors[start:end]
            if metric == "l2":
                diff = chunk - q
                scores = -np.einsum("ij,ij->i", diff, diff)
            elif metric == "cosine":
                scores = (chunk @ q) / (vector_norms[start:end] * q_norm)  # type: ignore[index]
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


def _write_tests_jsonl(
    query_rows: list[dict[str, Any]],
    knn_pos_scores: list[tuple[np.ndarray, np.ndarray]],
    id_mapping: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row, (pos, scores) in zip(query_rows, knn_pos_scores):
            out_row = dict(row)
            out_row["closest_ids"] = [int(id_mapping[int(i)]) for i in pos.tolist()]
            out_row["closest_scores"] = [float(s) for s in scores.tolist()]
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")


def _load_state(state_file: Path) -> tuple[int, int]:
    state = json.loads(state_file.read_text(encoding="utf-8"))
    base_id = int(state["base_id"])
    max_id = int(state["max_id"])
    if max_id < base_id:
        raise ValueError(f"invalid id range: base_id={base_id}, max_id={max_id}")
    return base_id, max_id


def write_live_tests_jsonl(
    *,
    dataset_name: str,
    state_file: Path,
    tests_out: Path,
    host: str,
    port: int,
    chunk_size: int = 5000,
) -> None:
    """内存中重算 KNN，只写 tests.jsonl。"""
    base_id, max_id = _load_state(state_file)
    cfg = load_dataset_config(dataset_name)
    query_rows = load_query_rows_from_benchmark(dataset_name)

    rows = _sorted_milvus_rows(base_id=base_id, max_id=max_id, host=host, port=port, chunk_size=chunk_size)
    vectors, source_ids = _corpus_arrays(rows)

    knn = _compute_knn_positions_for_queries(
        vectors,
        query_rows,
        distance=str(cfg.get("distance", "cosine")),
    )
    _write_tests_jsonl(query_rows, knn, source_ids, tests_out)
    print(f"Wrote live tests (KNN only): {tests_out}", flush=True)


def export_final_vectors_and_tests(
    *,
    dataset_name: str,
    state_file: Path,
    export_dir: Path,
    host: str,
    port: int,
    chunk_size: int = 5000,
) -> None:
    """收尾导出：仅 vectors.npy 与 tests.jsonl。"""
    base_id, max_id = _load_state(state_file)
    cfg = load_dataset_config(dataset_name)
    query_rows = load_query_rows_from_benchmark(dataset_name)

    rows = _sorted_milvus_rows(base_id=base_id, max_id=max_id, host=host, port=port, chunk_size=chunk_size)
    vectors, source_ids = _corpus_arrays(rows)

    export_dir.mkdir(parents=True, exist_ok=True)

    mmap = open_memmap(export_dir / "vectors.npy", mode="w+", dtype=np.float32, shape=vectors.shape)
    mmap[:] = vectors[:]
    del mmap

    knn = _compute_knn_positions_for_queries(
        vectors,
        query_rows,
        distance=str(cfg.get("distance", "cosine")),
    )
    _write_tests_jsonl(query_rows, knn, source_ids, export_dir / "tests.jsonl")

    meta_reload = {
        "base_id": base_id,
        "max_id": max_id,
        "num_vectors": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]),
        "source_dataset": dataset_name,
        "note_zh": (
            "漂移导出时 Milvus 实体 id 为连续整数 [base_id, max_id]；"
            "vectors.npy 第 i 行对应 id = base_id + i。"
            "重新跑 benchmark 上传时需将 tests.jsonl 的 closest_ids 映射为 0..N-1（见 prepare_final_export_for_benchmark.py）。"
        ),
    }
    (export_dir / "reload_meta.json").write_text(
        json.dumps(meta_reload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Final export: {export_dir / 'vectors.npy'}, {export_dir / 'tests.jsonl'}, reload_meta.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus 漂移与当前语料上的精确 KNN")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(str(MILVUS_DEFAULT_PORT)))
    parser.add_argument("--dataset", default="random-geo-radius-2048-angular-no-filters")
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="本轮插入向量条数",
    )
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="本轮删除的最旧向量条数（可与插入数量不同，例如多插少删使集合变大）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="若指定则插入与删除均使用该批量（等价于同时覆盖 insert/delete）",
    )
    parser.add_argument(
        "--drift-insert-dist",
        choices=("gaussian", "uniform_cube", "laplace", "sparse_orthogonal"),
        default="laplace",
        help="漂移插入向量采样分布（默认 laplace，与典型 Gaussian 基准向量差别明显）",
    )
    parser.add_argument("--base-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=99999)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--get-initial-count", metavar="DATASET", default=None)
    parser.add_argument(
        "--write-live-tests",
        metavar="TESTS_JSONL",
        default=None,
        help="按 state-file 从 Milvus 重算 KNN，仅写入该 tests.jsonl（不写 vectors.npy）",
    )
    parser.add_argument(
        "--export-final",
        metavar="DIR",
        default=None,
        help="收尾：将当前 Milvus 语料写入 DIR/vectors.npy 与 DIR/tests.jsonl",
    )
    args = parser.parse_args()

    if args.get_initial_count:
        print(get_initial_count(args.get_initial_count))
        return

    if args.stats_only:
        stats, num_entities = get_segment_vector_counts(host=args.host, port=args.port)
        sum_seg = sum(s["num_rows"] for s in stats)
        print(
            f"Segment 数: {len(stats)}, "
            f"各 segment num_rows 之和={sum_seg}, "
            f"collection.num_entities={num_entities}"
        )
        for i, s in enumerate(stats):
            print(
                f"  Segment {i + 1}: segment_id={s['segment_id']}, "
                f"num_rows={s['num_rows']}, state={s['state']}"
            )
        if args.output_stats:
            out = {
                "segments": stats,
                "segment_count": len(stats),
                "sum_segment_num_rows": sum_seg,
                "collection_num_entities": num_entities,
                "note_zh": (
                    "sum_segment_num_rows 为 Query Segment 上报行数之和，删除与 compaction 过程中常与逻辑实体数不一致；"
                    "集合真实向量条数以 collection_num_entities 为准。"
                ),
            }
            with open(args.output_stats, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        return

    if args.write_live_tests:
        if not args.state_file:
            raise SystemExit("--write-live-tests requires --state-file")
        write_live_tests_jsonl(
            dataset_name=args.dataset,
            state_file=Path(args.state_file),
            tests_out=Path(args.write_live_tests),
            host=args.host,
            port=args.port,
        )
        return

    if args.export_final:
        if not args.state_file:
            raise SystemExit("--export-final requires --state-file")
        export_final_vectors_and_tests(
            dataset_name=args.dataset,
            state_file=Path(args.state_file),
            export_dir=Path(args.export_final),
            host=args.host,
            port=args.port,
        )
        return

    base_id = args.base_id
    max_id = args.max_id
    if args.state_file and os.path.exists(args.state_file):
        with open(args.state_file, "r") as f:
            state = json.load(f)
            base_id = state["base_id"]
            max_id = state["max_id"]

    insert_bs = args.batch_size if args.batch_size is not None else args.insert_batch_size
    delete_bs = args.batch_size if args.batch_size is not None else args.delete_batch_size
    if insert_bs < 1 or delete_bs < 1:
        raise SystemExit("insert-batch-size 与 delete-batch-size 必须 >= 1")

    current_n = max_id - base_id + 1
    if delete_bs > current_n:
        raise SystemExit(
            f"delete-batch-size ({delete_bs}) 大于当前集合向量数 ({current_n})，请减小删除批量或先扩大初始数据"
        )

    new_base, new_max = run_drift_cycle(
        host=args.host,
        port=args.port,
        dataset_name=args.dataset,
        insert_batch_size=insert_bs,
        delete_batch_size=delete_bs,
        drift_insert_distribution=args.drift_insert_dist,
        base_id=base_id,
        max_id=max_id,
    )

    if args.state_file:
        with open(args.state_file, "w") as f:
            json.dump({"base_id": new_base, "max_id": new_max}, f, indent=2)

    print(
        f"Drift cycle done: inserted {insert_bs}, deleted {delete_bs}, "
        f"insert_dist={args.drift_insert_dist}"
    )
    print(f"New state: base_id={new_base}, max_id={new_max}")


if __name__ == "__main__":
    main()
