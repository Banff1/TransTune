#!/usr/bin/env python3
"""
数据漂移模拟脚本：向 Milvus 插入新向量并删除旧向量，模拟数据更新场景。
不修改原始数据集文件，所有新向量在内存中随机生成。

快照导出：可选用 --export-snapshot 将当前 Milvus corpus 导出到 auto-configure/vdtuner/adapt/drift_exports/…
并由 drift_snapshot.py 重算与原数据集相同的查询向量在新的 corpus 上的精确 KNN。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ADAPT_DIR = Path(__file__).resolve().parent
VDTUNER_DIR = ADAPT_DIR.parent
AUTO_CONFIGURE_ROOT = VDTUNER_DIR.parent
VDB_ROOT = AUTO_CONFIGURE_ROOT.parent
BENCHMARK_ROOT = VDB_ROOT / "vector-db-benchmark-master"

sys.path.insert(0, str(ADAPT_DIR))
sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
from pymilvus import Collection, connections, utility

from engine.clients.milvus.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)


def load_dataset_config(dataset_name: str) -> dict:
    """从 datasets.json 加载数据集配置"""
    datasets_path = BENCHMARK_ROOT / "datasets" / "datasets.json"
    with open(datasets_path, "r") as f:
        configs = json.load(f)
    for cfg in configs:
        if cfg["name"] == dataset_name:
            return cfg
    raise ValueError(f"Dataset '{dataset_name}' not found in datasets.json")


def get_initial_count(dataset_name: str) -> int:
    """从数据集文件推断初始向量数量（不修改原文件）"""
    cfg = load_dataset_config(dataset_name)
    base = BENCHMARK_ROOT / "datasets"
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

        h5_path = base / path_str
        with h5py.File(h5_path, "r") as f:
            return int(f["train"].shape[0])
    elif cfg["type"] == "jsonl":
        vectors_file = base / path_str / "vectors.jsonl"
        if vectors_file.exists():
            with open(vectors_file) as f:
                return sum(1 for _ in f)
    raise ValueError(f"Cannot infer count for dataset {dataset_name}")


def generate_random_vectors(n: int, dim: int, normalize: bool = True) -> np.ndarray:
    """生成随机向量，可选 L2 归一化（用于 cosine 距离）"""
    vectors = np.random.randn(n, dim).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def run_drift_cycle(
    host: str = "127.0.0.1",
    port: int = 19530,
    dataset_name: str = "random-geo-radius-2048-angular-no-filters",
    batch_size: int = 1000,
    base_id: int = 0,
    max_id: int = 99999,
) -> tuple[int, int]:
    """
    执行一轮数据漂移：插入 batch_size 个新向量，删除 batch_size 个最旧向量。
    返回新的 (base_id, max_id)。
    """
    cfg = load_dataset_config(dataset_name)
    dim = cfg["vector_size"]
    normalize = cfg.get("distance", "").lower() == "cosine"

    connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        host=host,
        port=str(port),
    )
    collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

    new_ids = list(range(max_id + 1, max_id + 1 + batch_size))
    vectors = generate_random_vectors(batch_size, dim, normalize)
    collection.insert([new_ids, vectors.tolist()])

    delete_ids = list(range(base_id, base_id + batch_size))
    ids_str = ",".join(str(i) for i in delete_ids)
    expr = f"id in [{ids_str}]"
    collection.delete(expr)

    collection.flush()

    connections.disconnect(MILVUS_DEFAULT_ALIAS)

    return base_id + batch_size, max_id + batch_size


def get_segment_vector_counts(
    host: str = "127.0.0.1",
    port: int = 19530,
) -> list[dict]:
    connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        host=host,
        port=str(port),
    )
    try:
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
        return result
    finally:
        connections.disconnect(MILVUS_DEFAULT_ALIAS)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行数据漂移：插入新向量、删除旧向量")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus 主机")
    parser.add_argument("--port", type=int, default=int(str(MILVUS_DEFAULT_PORT)), help="Milvus 端口")
    parser.add_argument(
        "--dataset",
        default="random-geo-radius-2048-angular-no-filters",
        help="数据集名称（用于获取 vector_size 和 distance）",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="每轮插入/删除的向量数量")
    parser.add_argument("--base-id", type=int, default=0, help="当前最旧未删除的 ID（由主脚本维护）")
    parser.add_argument("--max-id", type=int, default=99999, help="当前已插入的最大 ID（由主脚本维护）")
    parser.add_argument("--state-file", default=None, help="状态文件路径，用于保存/读取 base_id 和 max_id")
    parser.add_argument("--stats-only", action="store_true", help="仅获取各 segment 统计，不执行漂移")
    parser.add_argument("--output-stats", default=None, help="将 segment 统计写入此文件（JSON 格式）")
    parser.add_argument(
        "--get-initial-count",
        metavar="DATASET",
        default=None,
        help="仅输出指定数据集的初始向量数量并退出",
    )
    parser.add_argument(
        "--export-snapshot",
        metavar="DIR",
        default=None,
        help="将当前 Milvus corpus（由 state-file 划定 id 区间）导出为数据集快照并重算精确 KNN 到 DIR",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="不进行插删，仅按当前 state-file 导出快照（用于 cycle 0 基线或无漂移快照）",
    )
    parser.add_argument(
        "--export-cycle-note",
        default=None,
        help="写入 snapshot_meta.json 的备注（例如 cycle 编号说明）",
    )
    args = parser.parse_args()

    if args.get_initial_count:
        print(get_initial_count(args.get_initial_count))
        return

    if args.stats_only:
        stats = get_segment_vector_counts(host=args.host, port=args.port)
        total = sum(s["num_rows"] for s in stats)
        print(f"Segment 数量: {len(stats)}, 总向量数: {total}")
        for i, s in enumerate(stats):
            print(
                f"  Segment {i + 1}: segment_id={s['segment_id']}, "
                f"num_rows={s['num_rows']}, state={s['state']}"
            )
        if args.output_stats:
            out = {"segments": stats, "total_vectors": total, "segment_count": len(stats)}
            with open(args.output_stats, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        return

    if args.export_snapshot and args.export_only:
        state_path = Path(args.state_file) if args.state_file else None
        if state_path is None or not state_path.exists():
            raise SystemExit("export-only requires --state-file pointing to an existing JSON file.")
        from drift_snapshot import export_post_drift_snapshot

        ok, msg = export_post_drift_snapshot(
            dataset_name=args.dataset,
            state_file=state_path,
            export_dir=Path(args.export_snapshot),
            host=args.host,
            port=args.port,
            cycle_note=args.export_cycle_note,
        )
        if not ok:
            raise SystemExit(msg)
        print(f"Snapshot export OK: {msg}", flush=True)
        return

    base_id = args.base_id
    max_id = args.max_id
    state_path_export: Path | None = Path(args.state_file) if args.state_file else None

    if args.state_file and os.path.exists(args.state_file):
        with open(args.state_file, "r") as f:
            state = json.load(f)
            base_id = state["base_id"]
            max_id = state["max_id"]

    new_base, new_max = run_drift_cycle(
        host=args.host,
        port=args.port,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        base_id=base_id,
        max_id=max_id,
    )

    if args.state_file:
        with open(args.state_file, "w") as f:
            json.dump({"base_id": new_base, "max_id": new_max}, f, indent=2)
        state_path_export = Path(args.state_file)

    print(f"Drift cycle done: inserted {args.batch_size}, deleted {args.batch_size}")
    print(f"New state: base_id={new_base}, max_id={new_max}")

    if args.export_snapshot:
        if state_path_export is None or not state_path_export.exists():
            raise SystemExit("export-snapshot requires a valid --state-file after drift.")
        from drift_snapshot import export_post_drift_snapshot

        ok, msg = export_post_drift_snapshot(
            dataset_name=args.dataset,
            state_file=state_path_export,
            export_dir=Path(args.export_snapshot),
            host=args.host,
            port=args.port,
            cycle_note=args.export_cycle_note,
            extra_meta={"after_drift": True, "batch_size": args.batch_size},
        )
        if not ok:
            raise SystemExit(msg)
        print(f"Snapshot export OK: {msg}", flush=True)


if __name__ == "__main__":
    main()
