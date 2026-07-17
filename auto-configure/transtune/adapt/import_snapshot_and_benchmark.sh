#!/usr/bin/env bash
#
# 导入指定 drift snapshot（vectors.npy + source_ids.npy）到 Milvus，
# 并基于该 snapshot 的 tests.jsonl 运行 benchmark 搜索性能测试。
#
# 模式：
#   默认：导入 + 性能测试
#   --import-only：仅导入，不跑性能
#   --benchmark-only：仅跑性能，不重新导入
#
# 用法：
#   ./import_snapshot_and_benchmark.sh [--import-only|--benchmark-only] [--eval-dataset-name NAME] [SNAPSHOT_DIR] [SERVER_PATH] [ENGINE_NAME] [SOURCE_DATASET] [SERVER_HOST] [SERVER_PORT]
#
# 示例：
#   ./import_snapshot_and_benchmark.sh \
#     /path/to/TransTune/auto-configure/transtune/adapt/drift_exports/random-geo-radius-2048-angular-no-filters/20260513-214412/cycle_5 \
#     milvus-single-node milvus-p10 random-geo-radius-2048-angular-no-filters 127.0.0.1 19530

set -euo pipefail

ADAPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VDB_TUNING_ROOT=$(cd "$ADAPT_DIR/../../.." && pwd)
BENCHMARK_ROOT="$VDB_TUNING_ROOT/vector-db-benchmark-master"

MODE_IMPORT_ONLY=0
MODE_BENCHMARK_ONLY=0
EVAL_DATASET_NAME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --import-only)
            MODE_IMPORT_ONLY=1
            shift
            ;;
        --benchmark-only)
            MODE_BENCHMARK_ONLY=1
            shift
            ;;
        --eval-dataset-name)
            if [[ $# -lt 2 ]]; then
                echo "❌ --eval-dataset-name 需要一个值"
                exit 1
            fi
            EVAL_DATASET_NAME_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法:"
            echo "  $0 [--import-only|--benchmark-only] [--eval-dataset-name NAME] [SNAPSHOT_DIR] [SERVER_PATH] [ENGINE_NAME] [SOURCE_DATASET] [SERVER_HOST] [SERVER_PORT]"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [[ "$MODE_IMPORT_ONLY" -eq 1 && "$MODE_BENCHMARK_ONLY" -eq 1 ]]; then
    echo "❌ --import-only 与 --benchmark-only 不能同时使用"
    exit 1
fi

SNAPSHOT_DIR=${1:-"/path/to/TransTune/auto-configure/transtune/adapt/drift_exports/random-geo-radius-2048-angular-no-filters/20260513-214412/cycle_5"}
SERVER_PATH=${2:-"milvus-single-node"}
ENGINE_NAME=${3:-"milvus-p10"}
SOURCE_DATASET=${4:-"random-geo-radius-2048-angular-no-filters"}
SERVER_HOST=${5:-"127.0.0.1"}
SERVER_PORT=${6:-"19530"}
INSERT_BATCH_SIZE=${INSERT_BATCH_SIZE:-1000}

ENGINE_CONFIG_PATH="$BENCHMARK_ROOT/experiments/configurations/$SERVER_PATH.json"
DATASETS_JSON_PATH="$BENCHMARK_ROOT/datasets/datasets.json"
RUN_TS=$(date +%Y%m%d-%H%M%S)
if [[ -n "$EVAL_DATASET_NAME_OVERRIDE" ]]; then
    EVAL_DATASET_NAME="$EVAL_DATASET_NAME_OVERRIDE"
else
    EVAL_DATASET_NAME="${SOURCE_DATASET}-snapshot-eval-${RUN_TS}"
fi

VENV_PATH=${VENV_PATH:-"/path/to/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

if [ -f "$VENV_PATH/bin/python3.12" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3.12"
elif [ -f "$VENV_PATH/bin/python3" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3"
else
    PYTHON_CMD="python3.11"
fi

if [ ! -d "$SNAPSHOT_DIR" ]; then
    echo "❌ snapshot 目录不存在: $SNAPSHOT_DIR"
    exit 1
fi
if [ ! -f "$SNAPSHOT_DIR/vectors.npy" ] || [ ! -f "$SNAPSHOT_DIR/source_ids.npy" ] || [ ! -f "$SNAPSHOT_DIR/tests.jsonl" ]; then
    echo "❌ snapshot 缺少必要文件，要求包含 vectors.npy/source_ids.npy/tests.jsonl"
    exit 1
fi
if [ ! -f "$ENGINE_CONFIG_PATH" ]; then
    echo "❌ 未找到 engine 配置文件: $ENGINE_CONFIG_PATH"
    exit 1
fi

relative_path_from_datasets_root() {
    local abs_path="$1"
    "$PYTHON_CMD" - "$BENCHMARK_ROOT" "$abs_path" <<'PY'
import os
import sys

benchmark_root = sys.argv[1]
abs_path = sys.argv[2]
datasets_root = os.path.join(benchmark_root, "datasets")
print(os.path.relpath(abs_path, datasets_root))
PY
}

upsert_eval_dataset_config() {
    local eval_path_abs="$1"
    local eval_path_rel
    eval_path_rel="$(relative_path_from_datasets_root "$eval_path_abs")"

    "$PYTHON_CMD" - "$DATASETS_JSON_PATH" "$SOURCE_DATASET" "$EVAL_DATASET_NAME" "$eval_path_rel" <<'PY'
import json
import sys
from pathlib import Path

datasets_json = Path(sys.argv[1])
source_dataset = sys.argv[2]
eval_dataset = sys.argv[3]
eval_path_rel = sys.argv[4]

configs = json.loads(datasets_json.read_text(encoding="utf-8"))
source_cfg = next((c for c in configs if c.get("name") == source_dataset), None)
if source_cfg is None:
    raise SystemExit(f"source dataset not found in datasets.json: {source_dataset}")

configs = [c for c in configs if c.get("name") != eval_dataset]
new_cfg = {
    "name": eval_dataset,
    "vector_size": source_cfg["vector_size"],
    "distance": source_cfg["distance"],
    "type": "tar",
    "path": eval_path_rel,
}
if "schema" in source_cfg:
    new_cfg["schema"] = source_cfg["schema"]
configs.append(new_cfg)
datasets_json.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"updated dataset config: {eval_dataset} -> {eval_path_rel}")
PY
}

echo "======================================="
if [[ "$MODE_IMPORT_ONLY" -eq 1 ]]; then
    RUN_MODE_DESC="import-only"
elif [[ "$MODE_BENCHMARK_ONLY" -eq 1 ]]; then
    RUN_MODE_DESC="benchmark-only"
else
    RUN_MODE_DESC="import+benchmark"
fi

echo "📥 导入 snapshot 到 Milvus + 性能测试"
echo "Snapshot: $SNAPSHOT_DIR"
echo "Engine: $ENGINE_NAME | Source dataset: $SOURCE_DATASET"
echo "Host: $SERVER_HOST:$SERVER_PORT | Batch: $INSERT_BATCH_SIZE"
echo "Eval dataset name: $EVAL_DATASET_NAME"
echo "Mode: $RUN_MODE_DESC"
echo "======================================="

if [[ "$MODE_BENCHMARK_ONLY" -eq 0 ]]; then
echo ">>> [Step 1] 导入 snapshot 向量到 Milvus..."
"$PYTHON_CMD" - "$SNAPSHOT_DIR" "$ENGINE_CONFIG_PATH" "$ENGINE_NAME" "$SOURCE_DATASET" "$SERVER_HOST" "$SERVER_PORT" "$INSERT_BATCH_SIZE" "$BENCHMARK_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    wait_for_index_building_complete,
)

snapshot_dir = Path(sys.argv[1]).resolve()
engine_config_path = Path(sys.argv[2]).resolve()
engine_name = sys.argv[3]
source_dataset = sys.argv[4]
host = sys.argv[5]
port = str(sys.argv[6])
batch_size = int(sys.argv[7])
benchmark_root = Path(sys.argv[8]).resolve()

sys.path.insert(0, str(benchmark_root))

from engine.base_client.distances import Distance
from engine.clients.milvus.config import DISTANCE_MAPPING, MILVUS_COLLECTION_NAME, MILVUS_DEFAULT_ALIAS

datasets_json_path = benchmark_root / "datasets" / "datasets.json"
datasets_cfg = json.loads(datasets_json_path.read_text(encoding="utf-8"))
source_cfg = next((c for c in datasets_cfg if c.get("name") == source_dataset), None)
if source_cfg is None:
    raise SystemExit(f"source dataset not found in datasets.json: {source_dataset}")

distance = Distance.from_name(source_cfg["distance"])
metric_type = DISTANCE_MAPPING[distance]

engine_cfg_list = json.loads(engine_config_path.read_text(encoding="utf-8"))
engine_cfg = next((e for e in engine_cfg_list if e.get("name") == engine_name), None)
if engine_cfg is None:
    raise SystemExit(f"engine not found in {engine_config_path}: {engine_name}")

upload_params = engine_cfg.get("upload_params", {})
index_params = {
    "metric_type": metric_type,
    "index_type": upload_params.get("index_type", "HNSW"),
    "params": dict(upload_params.get("index_params", {})),
}

vectors = np.load(snapshot_dir / "vectors.npy", mmap_mode="r")
source_ids = np.load(snapshot_dir / "source_ids.npy", mmap_mode="r")
if vectors.shape[0] != source_ids.shape[0]:
    raise SystemExit(
        f"row count mismatch: vectors={vectors.shape[0]} source_ids={source_ids.shape[0]}"
    )
if vectors.ndim != 2:
    raise SystemExit(f"vectors.npy must be 2D, got shape={vectors.shape}")

row_count, dim = int(vectors.shape[0]), int(vectors.shape[1])
print(f"snapshot rows={row_count}, dim={dim}, metric={metric_type}, index={index_params['index_type']}")

connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host, port=port)
try:
    if utility.has_collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS):
        utility.drop_collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description=MILVUS_COLLECTION_NAME)

    handler = connections._fetch_handler(MILVUS_DEFAULT_ALIAS)
    handler.create_collection(MILVUS_COLLECTION_NAME, schema, shards_num=2)
    collection = Collection(name=MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)
        ids_batch = source_ids[start:end].astype(np.int64).tolist()
        vectors_batch = vectors[start:end].astype(np.float32).tolist()
        collection.insert([ids_batch, vectors_batch])
        if end % (batch_size * 20) == 0 or end == row_count:
            print(f"  inserted {end}/{row_count}")

    collection.flush()
    collection.create_index(field_name="vector", index_params=index_params)
    for index in collection.indexes:
        wait_for_index_building_complete(
            MILVUS_COLLECTION_NAME,
            index_name=index.index_name,
            using=MILVUS_DEFAULT_ALIAS,
        )
    collection.load()
    print("import + index build + load done")
finally:
    connections.disconnect(MILVUS_DEFAULT_ALIAS)
PY
else
    echo ">>> [Step 1] 跳过导入（benchmark-only 模式）"
fi

echo ">>> [Step 2] 注册 snapshot 数据集配置..."
upsert_eval_dataset_config "$SNAPSHOT_DIR"

if [[ "$MODE_IMPORT_ONLY" -eq 0 ]]; then
    echo ">>> [Step 3] 调用 run.py 执行性能测试（跳过上传）..."
    cd "$BENCHMARK_ROOT"
    export no_proxy="localhost,127.0.0.1,::1"
    "$PYTHON_CMD" run.py --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST" --skip-upload
else
    echo ">>> [Step 3] 跳过性能测试（import-only 模式）"
fi

LATEST_RESULT=$(ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${EVAL_DATASET_NAME}.*search" | head -n 1 || true)

echo "======================================="
echo "✅ 完成"
echo "Eval dataset: $EVAL_DATASET_NAME"
if [[ "$MODE_IMPORT_ONLY" -eq 1 ]]; then
    echo "ℹ️  import-only 已完成。后续仅测性能可执行："
    echo "   $0 --benchmark-only --eval-dataset-name \"$EVAL_DATASET_NAME\" \"$SNAPSHOT_DIR\" \"$SERVER_PATH\" \"$ENGINE_NAME\" \"$SOURCE_DATASET\" \"$SERVER_HOST\" \"$SERVER_PORT\""
elif [ -n "$LATEST_RESULT" ]; then
    echo "Latest search result: $BENCHMARK_ROOT/results/$LATEST_RESULT"
else
    echo "⚠️ 未找到对应 search 结果文件，请检查 run.py 输出。"
fi
echo "======================================="
