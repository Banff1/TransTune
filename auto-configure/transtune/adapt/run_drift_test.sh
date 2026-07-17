#!/usr/bin/env bash
#
# 数据漂移性能测试脚本
# 验证随着数据插入和删除（数据漂移），向量数据库原有 knob 配置是否出现性能下降。
#
# 流程：
# 1. 仅导入指定数据集（不搜索）
# 2. 导出 cycle 0 快照并重算 KNN，基于 cycle 0 的 tests.jsonl 评测初始性能
# 3. 每轮先漂移（插入/删除），再导出当轮快照并重算 KNN，然后用当轮 tests.jsonl 评测性能
# 4. 所有漂移完成后，再导出 final 快照
#
# 注意：不修改原始数据集文件，新向量在内存中随机生成
#
# 用法：
#   /path/to/adapt/run_drift_test.sh [SERVER_PATH] [ENGINE_NAME] [DATASET] [BATCH_SIZE] [NUM_CYCLES]
#
# 示例：
#   ./run_drift_test.sh milvus-single-node milvus-p10 random-geo-radius-2048-angular-no-filters 1000 10
#

set -e

ADAPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VDB_TUNING_ROOT=$(cd "$ADAPT_DIR/../../.." && pwd)
BENCHMARK_ROOT="$VDB_TUNING_ROOT/vector-db-benchmark-master"

SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
DATASETS=${3:-"random-geo-radius-2048-angular-no-filters"}
BATCH_SIZE=${4:-1000}
NUM_CYCLES=${5:-10}
SERVER_HOST="127.0.0.1"

MILVUS_DIR="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH"
MONITOR_DIR="$BENCHMARK_ROOT/monitoring"
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"
DRIFT_STATE_FILE="$ADAPT_DIR/.drift_state.json"
DRIFT_SEGMENT_STATS_ROOT="$BENCHMARK_ROOT/results/drift_segment_stats"
DATASETS_JSON_PATH="$BENCHMARK_ROOT/datasets/datasets.json"

RUN_TS=$(date +%Y%m%d-%H%M%S)
DRIFT_EXPORT_RUN_DIR="$ADAPT_DIR/drift_exports/$DATASETS/$RUN_TS"
DRIFT_METRICS_ARCHIVE_DIR="$BENCHMARK_ROOT/results/drift_cycle_metrics/$RUN_TS"
DRIFT_METRICS_SUMMARY_FILE="$DRIFT_METRICS_ARCHIVE_DIR/metrics_summary.jsonl"
DRIFT_RESULTS_FILE="$DRIFT_METRICS_ARCHIVE_DIR/drift_test_results.jsonl"
DRIFT_SEGMENT_STATS_DIR="$DRIFT_SEGMENT_STATS_ROOT/$RUN_TS"
EVAL_DATASET_NAME="${DATASETS}-drift-live-eval-${RUN_TS}"

# 虚拟环境
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

echo "======================================="
echo "📊 数据漂移性能测试"
echo "Engine: $ENGINE_NAME | Dataset: $DATASETS"
echo "Batch: $BATCH_SIZE 向量/轮 | 漂移轮数: $NUM_CYCLES"
echo "漂移快照: 每轮导出 → $DRIFT_EXPORT_RUN_DIR/cycle_*"
echo "最终导出: $DRIFT_EXPORT_RUN_DIR/final"
echo "======================================="

archive_cycle_result() {
    local cycle="$1"
    local src_name="$2"
    local src_path="$BENCHMARK_ROOT/results/$src_name"

    if [ ! -f "$src_path" ]; then
        echo "    ⚠️  [Cycle $cycle] 未找到结果文件: $src_path"
        return 1
    fi

    mkdir -p "$DRIFT_METRICS_ARCHIVE_DIR"
    local archived_name="cycle_${cycle}-${src_name}"
    local archived_rel="drift_cycle_metrics/$RUN_TS/$archived_name"
    local archived_path="$BENCHMARK_ROOT/results/$archived_rel"
    cp "$src_path" "$archived_path"

    # 记录稳定归档路径，避免后续 benchmark 结果覆盖同名文件。
    echo "{\"cycle\": $cycle, \"file\": \"$archived_rel\"}" >> "$DRIFT_RESULTS_FILE"

    "$PYTHON_CMD" - "$cycle" "$src_name" "$archived_rel" "$archived_path" <<'PY' >> "$DRIFT_METRICS_SUMMARY_FILE"
import json
import sys
from pathlib import Path

cycle = int(sys.argv[1])
source_file = sys.argv[2]
archived_rel = sys.argv[3]
archived_path = Path(sys.argv[4])
payload = json.loads(archived_path.read_text(encoding="utf-8"))
results = payload.get("results", {})

out = {
    "cycle": cycle,
    "source_file": source_file,
    "archived_file": archived_rel,
    "rps": float(results.get("rps", 0.0) or 0.0),
    "mean_precisions": float(results.get("mean_precisions", 0.0) or 0.0),
    "recalls": float(results.get("recalls", 0.0) or 0.0),
}
print(json.dumps(out, ensure_ascii=False))
PY

    return 0
}

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

    "$PYTHON_CMD" - "$DATASETS_JSON_PATH" "$DATASETS" "$EVAL_DATASET_NAME" "$eval_path_rel" <<'PY'
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

latest_search_result_for_dataset() {
    local dataset_name="$1"
    ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${dataset_name}.*search" | head -n 1
}

# 1. 启动监控
echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控进程 PID: $MONITOR_PID"
else
    echo "⚠️  未找到监控脚本，跳过。"
fi

# 2. 重置 Milvus
echo ">>> [Step 2] 重置 Milvus..."
cd "$MILVUS_DIR"
docker-compose down -v
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

docker-compose up -d

# 3. 等待服务就绪
echo ">>> [Step 3] 等待 Milvus 启动 (最多 120s)..."
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
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    [ $((WAIT_COUNT % 10)) -eq 0 ] && echo "    等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "    ⚠️  等待超时，继续尝试..."
fi

mkdir -p "$DRIFT_EXPORT_RUN_DIR"

# 4. 初始导入 + cycle 0 导出 + 基线性能评测
echo ">>> [Step 4] 导入数据并评测 cycle 0 ..."
cd "$BENCHMARK_ROOT"
export no_proxy="localhost,127.0.0.1,::1"

$PYTHON_CMD run.py --engines "$ENGINE_NAME" --datasets "$DATASETS" --host "$SERVER_HOST" --skip-search

INITIAL_COUNT=$($PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" --get-initial-count "$DATASETS")
echo "    初始向量数: $INITIAL_COUNT"

echo "{\"cycle\":0,\"base_id\":0,\"max_id\":$((INITIAL_COUNT - 1))}" > "$DRIFT_STATE_FILE"

mkdir -p "$BENCHMARK_ROOT/results" "$DRIFT_SEGMENT_STATS_DIR"
rm -f "$DRIFT_RESULTS_FILE"
mkdir -p "$DRIFT_METRICS_ARCHIVE_DIR"
rm -f "$DRIFT_METRICS_SUMMARY_FILE"

echo "    [Cycle 0] 统计各 segment 向量数量..."
$PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" --stats-only \
    --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_0.json" 2>/dev/null || true

echo "    [Cycle 0] 导出基线快照并重算 KNN..."
$PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" \
    --dataset "$DATASETS" \
    --state-file "$DRIFT_STATE_FILE" \
    --host "$SERVER_HOST" \
    --export-snapshot "$DRIFT_EXPORT_RUN_DIR/cycle_0" \
    --export-only \
    --export-cycle-note "cycle_0_baseline_after_initial_upload"

upsert_eval_dataset_config "$DRIFT_EXPORT_RUN_DIR/cycle_0"

echo "    [Cycle 0] 使用重算 KNN 的 tests.jsonl 评测初始性能..."
$PYTHON_CMD "$BENCHMARK_ROOT/run.py" --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST" --skip-upload
RES_FILE=$(latest_search_result_for_dataset "$EVAL_DATASET_NAME")
if [ -n "$RES_FILE" ]; then
    archive_cycle_result 0 "$RES_FILE" || true
else
    echo "    ⚠️  [Cycle 0] 未找到性能结果文件。"
fi

# 5. 漂移循环
echo ">>> [Step 5] 开始数据漂移循环 ($NUM_CYCLES 轮)..."

for i in $(seq 1 "$NUM_CYCLES"); do
    echo ""
    echo "--- 漂移轮次 $i/$NUM_CYCLES ---"

    echo "    [${i}] 插入 $BATCH_SIZE 新向量，删除 $BATCH_SIZE 旧向量..."
    $PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" \
        --dataset "$DATASETS" \
        --batch-size "$BATCH_SIZE" \
        --state-file "$DRIFT_STATE_FILE" \
        --host "$SERVER_HOST" \
        --export-snapshot "$DRIFT_EXPORT_RUN_DIR/cycle_${i}" \
        --export-cycle-note "cycle_${i}_after_drift"

    upsert_eval_dataset_config "$DRIFT_EXPORT_RUN_DIR/cycle_${i}"

    sleep 5

    echo "    [${i}] 使用重算 KNN 的 tests.jsonl 运行性能测试..."
    $PYTHON_CMD "$BENCHMARK_ROOT/run.py" --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST" --skip-upload

    RES_FILE=$(latest_search_result_for_dataset "$EVAL_DATASET_NAME")
    if [ -n "$RES_FILE" ]; then
        archive_cycle_result "$i" "$RES_FILE" || true
    else
        echo "    ⚠️  [Cycle $i] 未找到性能结果文件。"
    fi

    echo "    [${i}] 统计各 segment 向量数量..."
    $PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" --stats-only \
        --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_${i}.json" 2>/dev/null || true
done

# 6. 收尾
echo ""
echo ">>> [Step 6] 收尾..."

echo "    导出 final 快照（全部漂移结束后的向量与测试集）..."
$PYTHON_CMD "$ADAPT_DIR/run_drift_cycle.py" \
    --dataset "$DATASETS" \
    --state-file "$DRIFT_STATE_FILE" \
    --host "$SERVER_HOST" \
    --export-snapshot "$DRIFT_EXPORT_RUN_DIR/final" \
    --export-only \
    --export-cycle-note "final_after_${NUM_CYCLES}_drift_cycles"

if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
    mkdir -p "$MONITOR_DIR/results"
    LOG_NAME=$(echo "$ENGINE_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')
    mv "$MONITOR_DIR/docker.stats.jsonl" "$MONITOR_DIR/results/${LOG_NAME}-drift-docker.stats.jsonl" 2>/dev/null || true
fi

echo ""
echo "======================================="
echo "📊 数据漂移测试完成"
echo "结果文件: $DRIFT_RESULTS_FILE"
echo "性能指标归档: $DRIFT_METRICS_ARCHIVE_DIR/cycle_*.json"
echo "指标汇总(JSONL): $DRIFT_METRICS_SUMMARY_FILE"
echo "漂移快照(KNN): $DRIFT_EXPORT_RUN_DIR/cycle_*"
echo "最终导出(KNN): $DRIFT_EXPORT_RUN_DIR/final"
echo "Segment 统计: $DRIFT_SEGMENT_STATS_DIR/cycle_*.json"
echo "======================================="
