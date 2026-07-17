#!/usr/bin/env bash
#
# 使用 final_export 目录中的 vectors.npy + tests.jsonl，仿照 vector-db-benchmark-master/run_engine_test.sh：
#   重置 Milvus → 注册临时数据集 → run.py 全量上传并搜索 → 归档结果到本目录。
#
# 用法:
#   ./run_final_export_benchmark.sh [FINAL_EXPORT路径] [SOURCE_DATASET] [SERVER_PATH] [ENGINE_NAME]
#
# FINAL_EXPORT 可为相对于本脚本目录的路径（如 final_export/20260514-195755）或绝对路径。
# SOURCE_DATASET 用于从 datasets.json 复制 vector_size / distance / schema（与漂移时基准集一致）。
#
# 可选环境变量:
#   EXPORT_BASE_ID  — 若导出目录无 reload_meta.json，且 .drift_state.json 与向量条数不匹配时，手动指定漂移最小 Milvus id。
#
# 示例:
#   ./run_final_export_benchmark.sh final_export/20260514-195755 random-geo-radius-2048-angular-no-filters milvus-single-node milvus-p10
#

set -e

DRIFTING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VDB_TUNING_ROOT=$(cd "$DRIFTING_DIR/../../../.." && pwd)
BENCHMARK_ROOT="$VDB_TUNING_ROOT/vector-db-benchmark-master"

EXPORT_ARG=${1:-"final_export/20260514-195755"}
SOURCE_DATASET=${2:-"random-geo-radius-2048-angular-no-filters"}
SERVER_PATH=${3:-"milvus-single-node"}
ENGINE_NAME=${4:-"milvus-p10"}
SERVER_HOST="127.0.0.1"

if [[ "$EXPORT_ARG" = /* ]]; then
    EXPORT_ABS="$EXPORT_ARG"
else
    EXPORT_ABS="$DRIFTING_DIR/$EXPORT_ARG"
fi

if [ ! -d "$EXPORT_ABS" ]; then
    echo "❌ 导出目录不存在: $EXPORT_ABS"
    exit 1
fi
if [ ! -f "$EXPORT_ABS/vectors.npy" ] || [ ! -f "$EXPORT_ABS/tests.jsonl" ]; then
    echo "❌ 目录内需要 vectors.npy 与 tests.jsonl: $EXPORT_ABS"
    exit 1
fi

MILVUS_DIR="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH"
MONITOR_DIR="$BENCHMARK_ROOT/monitoring"
DATASETS_JSON_PATH="$BENCHMARK_ROOT/datasets/datasets.json"
DRIFT_STATE_FILE="$DRIFTING_DIR/.drift_state.json"

DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"

RUN_TS=$(date +%Y%m%d-%H%M%S)
RUN_TAG="$(basename "$EXPORT_ABS")-${RUN_TS}"
STAGING_DIR="$DRIFTING_DIR/benchmark_from_final/${RUN_TAG}"
EVAL_DATASET_NAME="${SOURCE_DATASET}-finalreload-${RUN_TAG}"
RESULTS_ARCHIVE="$DRIFTING_DIR/final_benchmark_results/${RUN_TAG}"

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

if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
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

upsert_benchmark_dataset() {
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
print(f"registered dataset: {eval_dataset} -> {eval_path_rel}")
PY
}

echo "======================================="
echo "final_export → Milvus → Benchmark"
echo "导出目录: $EXPORT_ABS"
echo "临时数据集名: $EVAL_DATASET_NAME"
echo "Engine: $ENGINE_NAME | 模板数据集: $SOURCE_DATASET"
echo "staging: $STAGING_DIR"
echo "======================================="

echo ">>> [Prepare] 重映射 tests.jsonl（Milvus id → 行号 0..N-1）..."
mkdir -p "$STAGING_DIR"
PREPARE_ARGS=(
    "$DRIFTING_DIR/prepare_final_export_for_benchmark.py"
    --export-dir "$EXPORT_ABS"
    --out-dir "$STAGING_DIR"
)
if [ -n "${EXPORT_BASE_ID:-}" ]; then
    PREPARE_ARGS+=(--base-id "$EXPORT_BASE_ID")
fi
if [ -f "$DRIFT_STATE_FILE" ]; then
    PREPARE_ARGS+=(--drift-state "$DRIFT_STATE_FILE")
fi
"$PYTHON_CMD" "${PREPARE_ARGS[@]}"

upsert_benchmark_dataset "$STAGING_DIR"

echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控 PID: $MONITOR_PID"
else
    echo "⚠️  未找到 monitor_docker.sh，跳过。"
fi

echo ">>> [Step 2] 重置 Milvus..."
cd "$MILVUS_DIR"
$COMPOSE_CMD down -v
sleep 5

CLEAN_HOST_VOLUMES=${CLEAN_HOST_VOLUMES:-1}
if [ "$CLEAN_HOST_VOLUMES" = "1" ]; then
    DEFAULT_VOLUME_ROOT="${DOCKER_VOLUME_DIRECTORY}/volumes"
    VOLUME_ROOT="${VOLUME_ROOT:-$DEFAULT_VOLUME_ROOT}"
    VOLUME_ROOT_ABS="$(realpath -m "$VOLUME_ROOT" 2>/dev/null || echo "")"
    DEFAULT_VOLUME_ROOT_ABS="$(realpath -m "$DEFAULT_VOLUME_ROOT" 2>/dev/null || echo "$DEFAULT_VOLUME_ROOT")"
    LEGACY_VOLUME_ROOT="$MILVUS_DIR/volumes"
    LEGACY_VOLUME_ROOT_ABS="$(realpath -m "$LEGACY_VOLUME_ROOT" 2>/dev/null || echo "")"
    if [ -n "$VOLUME_ROOT_ABS" ] && {
        [ "$VOLUME_ROOT_ABS" = "$DEFAULT_VOLUME_ROOT_ABS" ] ||
        [[ "$VOLUME_ROOT_ABS" == */milvus-single-node/volumes ]] ||
        { [ -n "$LEGACY_VOLUME_ROOT_ABS" ] && [ "$VOLUME_ROOT_ABS" = "$LEGACY_VOLUME_ROOT_ABS" ]; };
    }; then
        echo "    🧹 清理宿主机数据目录: $VOLUME_ROOT_ABS"
        sudo rm -rf "${VOLUME_ROOT_ABS:?}/"*
    fi
fi

$COMPOSE_CMD up -d

echo ">>> [Step 3] 等待 Milvus 就绪 (最多 120s)..."
MAX_WAIT=120
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if ! docker ps | grep -q milvus-standalone; then
        echo "    ❌ milvus-standalone 未运行"
        exit 1
    fi
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "    ✅ Milvus 已就绪"
        break
    fi
    STATUS=$(docker inspect --format='{{.State.Status}}' milvus-standalone 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "exited" ]; then
        echo "    ❌ 容器已退出"
        docker logs milvus-standalone --tail 40 2>&1 || true
        exit 1
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    [ $((WAIT_COUNT % 10)) -eq 0 ] && echo "    等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "    ⚠️  等待超时，继续尝试 benchmark..."
fi

echo ">>> [Step 4] 运行 run.py（上传 + 搜索）..."
cd "$BENCHMARK_ROOT"
export no_proxy="localhost,127.0.0.1,::1"
$PYTHON_CMD run.py --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST"

echo ">>> [Step 5] 收尾..."
mkdir -p "$RESULTS_ARCHIVE"
RES_SEARCH=$(ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${EVAL_DATASET_NAME}.*search" | head -n 1 || true)
RES_UPLOAD=$(ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${EVAL_DATASET_NAME}.*upload" | head -n 1 || true)
if [ -n "$RES_SEARCH" ]; then
    cp "$BENCHMARK_ROOT/results/$RES_SEARCH" "$RESULTS_ARCHIVE/"
    echo "    已归档 search 结果: $RESULTS_ARCHIVE/$RES_SEARCH"
fi
if [ -n "$RES_UPLOAD" ]; then
    cp "$BENCHMARK_ROOT/results/$RES_UPLOAD" "$RESULTS_ARCHIVE/"
    echo "    已归档 upload 结果: $RESULTS_ARCHIVE/$RES_UPLOAD"
fi

if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    mkdir -p "$DRIFTING_DIR/monitoring_results"
    LOG_NAME=$(echo "$ENGINE_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')
    if [ -f "$MONITOR_DIR/docker.stats.jsonl" ]; then
        mv "$MONITOR_DIR/docker.stats.jsonl" "$DRIFTING_DIR/monitoring_results/${LOG_NAME}-final-export-docker.stats-${RUN_TS}.jsonl" 2>/dev/null || true
    fi
fi

echo ""
echo "======================================="
echo "📊 完成"
echo "归档目录: $RESULTS_ARCHIVE"
if [ -n "$RES_SEARCH" ]; then
    RES_PATH="$BENCHMARK_ROOT/results/$RES_SEARCH"
    echo "查看 precision / rps:"
    "$PYTHON_CMD" -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding='utf-8'))
r = data.get('results', {})
print('  mean_precisions:', r.get('mean_precisions'))
print('  rps:', r.get('rps'))
print('  p95_time:', r.get('p95_time'))
" "$RES_PATH"
fi
echo "======================================="
