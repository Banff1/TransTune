#!/usr/bin/env python3
"""
将 final_export 目录（vectors.npy + tests.jsonl）转为可直接被 vector-db-benchmark 上传的格式。

漂移导出里 closest_ids 存的是 Milvus 实体 id（连续区间 [base_id, max_id]）；
benchmark 按 vectors.npy 行顺序上传，Record.id = 0..N-1，因此必须把 closest_ids 映射为 行下标。

优先使用同目录下 reload_meta.json 的 base_id；否则使用 --base-id 或与 vectors 行数匹配的 .drift_state.json。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _resolve_base_id(
    *,
    export_dir: Path,
    n_vectors: int,
    base_id_arg: int | None,
    drift_state_path: Path | None,
) -> int:
    meta_path = export_dir / "reload_meta.json"
    if base_id_arg is not None:
        return int(base_id_arg)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bn = int(meta.get("num_vectors", n_vectors))
        if bn != n_vectors:
            raise SystemExit(f"reload_meta.json num_vectors ({bn}) != vectors.npy rows ({n_vectors})")
        return int(meta["base_id"])
    if drift_state_path is not None and drift_state_path.exists():
        st = json.loads(drift_state_path.read_text(encoding="utf-8"))
        b = int(st["base_id"])
        m = int(st["max_id"])
        if m - b + 1 != n_vectors:
            raise SystemExit(
                f".drift_state.json 中 max_id-base_id+1={m - b + 1} 与 vectors.npy 行数 {n_vectors} 不一致；"
                "请改用导出目录下的 reload_meta.json（重新跑一遍 export-final）或显式传入 --base-id"
            )
        return b
    raise SystemExit(
        "无法确定 base_id：请在导出目录提供 reload_meta.json，或传 --base-id，或传匹配的 --drift-state"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="final_export → benchmark 数据集目录（重映射 id）")
    parser.add_argument("--export-dir", type=Path, required=True, help="含 vectors.npy、tests.jsonl 的 final_export 目录")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录（写入 vectors.npy、tests.jsonl）")
    parser.add_argument("--base-id", type=int, default=None, help="Milvus 最小实体 id（与第 0 行向量对应）")
    parser.add_argument(
        "--drift-state",
        type=Path,
        default=None,
        help="漂移状态 JSON（默认尝试脚本上级目录 another-version/.drift_state.json）",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    vectors_path = export_dir / "vectors.npy"
    tests_src = export_dir / "tests.jsonl"
    if not vectors_path.exists():
        raise SystemExit(f"缺少 {vectors_path}")
    if not tests_src.exists():
        raise SystemExit(f"缺少 {tests_src}")

    n = int(np.load(vectors_path, mmap_mode="r").shape[0])
    drift_state = args.drift_state
    if drift_state is None:
        candidate = export_dir.parent.parent / ".drift_state.json"
        if candidate.exists():
            drift_state = candidate

    base_id = _resolve_base_id(
        export_dir=export_dir,
        n_vectors=n,
        base_id_arg=args.base_id,
        drift_state_path=drift_state,
    )
    max_id = base_id + n - 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vectors_path, out_dir / "vectors.npy")

    out_tests = out_dir / "tests.jsonl"
    with tests_src.open(encoding="utf-8") as inf, out_tests.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            old_ids = row["closest_ids"]
            new_ids: list[int] = []
            for x in old_ids:
                oid = int(x)
                if oid < base_id or oid > max_id:
                    raise SystemExit(
                        f"tests.jsonl 中 id={oid} 不在导出区间 [{base_id}, {max_id}]，请检查 base_id 或导出是否匹配"
                    )
                new_ids.append(oid - base_id)
            row["closest_ids"] = new_ids
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"OK: {out_dir}")
    print(f"  base_id={base_id}, max_id={max_id}, num_vectors={n}")


if __name__ == "__main__":
    main()
