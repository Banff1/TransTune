#!/usr/bin/env bash
#
# another-version：每轮漂移后可插入条数与删除条数不同（默认多插少删使集合变大，更易观察到吞吐下降）。
# 漂移插入向量分布由 DRIFT_INSERT_DIST 控制，默认 laplace，与基准常见 Gaussian 稠密向量刻意不一致。
#
# 用法:
#   ./run_drift_test.sh [SERVER_PATH] [ENGINE_NAME] [DATASET] [INSERT_BATCH] [DELETE_BATCH] [NUM_CYCLES]
# 可选环境变量:
#   DRIFT_INSERT_DIST — gaussian | uniform_cube | laplace | sparse_orthogonal （默认 laplace）
#
# 示例:
#   ./run_drift_test.sh milvus-single-node milvus-p10 random-geo-radius-2048-angular-no-filters 6000 1000 8
#

set -e

DRIFTING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VDB_TUNING_ROOT=$(cd "$DRIFTING_DIR/../../../.." && pwd)
BENCHMARK_ROOT="$VDB_TUNING_ROOT/vector-db-benchmark-master"

SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
DATASETS=${3:-"random-geo-radius-2048-angular-no-filters"}
INSERT_BATCH=${4:-4000}
DELETE_BATCH=${5:-1000}
NUM_CYCLES=${6:-10}
DRIFT_INSERT_DIST=${DRIFT_INSERT_DIST:-laplace}
SERVER_HOST="127.0.0.1"

MILVUS_DIR="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH"
MONITOR_DIR="$BENCHMARK_ROOT/monitoring"
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"
DRIFT_STATE_FILE="$DRIFTING_DIR/.drift_state.json"
DATASETS_JSON_PATH="$BENCHMARK_ROOT/datasets/datasets.json"

RUN_TS=$(date +%Y%m%d-%H%M%S)
LIVE_EVAL_DIR="$DRIFTING_DIR/live_eval"
FINAL_EXPORT_DIR="$DRIFTING_DIR/final_export/$RUN_TS"
DRIFT_METRICS_ARCHIVE_DIR="$DRIFTING_DIR/drift_cycle_metrics/$RUN_TS"
DRIFT_METRICS_SUMMARY_FILE="$DRIFT_METRICS_ARCHIVE_DIR/metrics_summary.jsonl"
DRIFT_RESULTS_FILE="$DRIFT_METRICS_ARCHIVE_DIR/drift_test_results.jsonl"
DRIFT_SEGMENT_STATS_DIR="$DRIFTING_DIR/drift_segment_stats/$RUN_TS"
EVAL_DATASET_NAME="${DATASETS}-another-version-live-${RUN_TS}"

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
echo "📊 数据漂移性能测试 (another-version)"
echo "Engine: $ENGINE_NAME | Dataset: $DATASETS"
echo "漂移: 每轮插入 $INSERT_BATCH / 删除 $DELETE_BATCH | 轮数: $NUM_CYCLES"
echo "漂移插入分布 DRIFT_INSERT_DIST=$DRIFT_INSERT_DIST（与基准 Gaussian 稠密设定刻意区分）"
echo "每轮净增向量约 $((INSERT_BATCH - DELETE_BATCH))，累计约 +$(((INSERT_BATCH - DELETE_BATCH) * NUM_CYCLES))（不计 cycle 0）"
echo "运行时仅更新: $LIVE_EVAL_DIR/tests.jsonl"
echo "收尾导出: $FINAL_EXPORT_DIR/{vectors.npy,tests.jsonl}"
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
    local archived_path="$DRIFT_METRICS_ARCHIVE_DIR/$archived_name"
    cp "$src_path" "$archived_path"

    echo "{\"cycle\": $cycle, \"file\": \"$archived_name\", \"dir\": \"drift_cycle_metrics/$RUN_TS\"}" >> "$DRIFT_RESULTS_FILE"

    "$PYTHON_CMD" - "$cycle" "$src_name" "$archived_name" "$archived_path" <<'PY' >> "$DRIFT_METRICS_SUMMARY_FILE"
import json
import sys
from pathlib import Path

cycle = int(sys.argv[1])
source_file = sys.argv[2]
archived_name = sys.argv[3]
archived_path = Path(sys.argv[4])
payload = json.loads(archived_path.read_text(encoding="utf-8"))
results = payload.get("results", {})

out = {
    "cycle": cycle,
    "source_file": source_file,
    "archived_file": archived_name,
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

refresh_live_tests() {
    echo "    重算 KNN → $LIVE_EVAL_DIR/tests.jsonl（不写 vectors.npy）..."
    $PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" \
        --dataset "$DATASETS" \
        --state-file "$DRIFT_STATE_FILE" \
        --host "$SERVER_HOST" \
        --write-live-tests "$LIVE_EVAL_DIR/tests.jsonl"
}

echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控进程 PID: $MONITOR_PID"
else
    echo "⚠️  未找到监控脚本，跳过。"
fi

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

mkdir -p "$LIVE_EVAL_DIR"

echo ">>> [Step 4] 导入数据并评测 cycle 0 ..."
cd "$BENCHMARK_ROOT"
export no_proxy="localhost,127.0.0.1,::1"

$PYTHON_CMD run.py --engines "$ENGINE_NAME" --datasets "$DATASETS" --host "$SERVER_HOST" --skip-search

INITIAL_COUNT=$($PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --get-initial-count "$DATASETS")
echo "    初始向量数: $INITIAL_COUNT"

echo "{\"cycle\":0,\"base_id\":0,\"max_id\":$((INITIAL_COUNT - 1))}" > "$DRIFT_STATE_FILE"

mkdir -p "$BENCHMARK_ROOT/results" "$DRIFT_SEGMENT_STATS_DIR"
rm -f "$DRIFT_RESULTS_FILE"
mkdir -p "$DRIFT_METRICS_ARCHIVE_DIR"
rm -f "$DRIFT_METRICS_SUMMARY_FILE"

echo "    [Cycle 0] 统计各 segment 向量数量..."
$PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --stats-only \
    --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_0.json" 2>/dev/null || true

refresh_live_tests
upsert_eval_dataset_config "$LIVE_EVAL_DIR"

echo "    [Cycle 0] 搜索评测（skip-upload，precision 对应当前语料 KNN）..."
$PYTHON_CMD "$BENCHMARK_ROOT/run.py" --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST" --skip-upload
RES_FILE=$(latest_search_result_for_dataset "$EVAL_DATASET_NAME")
if [ -n "$RES_FILE" ]; then
    archive_cycle_result 0 "$RES_FILE" || true
else
    echo "    ⚠️  [Cycle 0] 未找到性能结果文件。"
fi

echo ">>> [Step 5] 漂移循环 ($NUM_CYCLES 轮)..."

for i in $(seq 1 "$NUM_CYCLES"); do
    echo ""
    echo "--- 漂移轮次 $i/$NUM_CYCLES ---"

    echo "    [${i}] 插入 $INSERT_BATCH / 删除 $DELETE_BATCH（dist=$DRIFT_INSERT_DIST）..."
    $PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" \
        --dataset "$DATASETS" \
        --insert-batch-size "$INSERT_BATCH" \
        --delete-batch-size "$DELETE_BATCH" \
        --drift-insert-dist "$DRIFT_INSERT_DIST" \
        --state-file "$DRIFT_STATE_FILE" \
        --host "$SERVER_HOST"

    sleep 5

    refresh_live_tests

    echo "    [${i}] 搜索评测（skip-upload）..."
    $PYTHON_CMD "$BENCHMARK_ROOT/run.py" --engines "$ENGINE_NAME" --datasets "$EVAL_DATASET_NAME" --host "$SERVER_HOST" --skip-upload

    RES_FILE=$(latest_search_result_for_dataset "$EVAL_DATASET_NAME")
    if [ -n "$RES_FILE" ]; then
        archive_cycle_result "$i" "$RES_FILE" || true
    else
        echo "    ⚠️  [Cycle $i] 未找到性能结果文件。"
    fi

    echo "    [${i}] segment 统计..."
    $PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --stats-only \
        --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_${i}.json" 2>/dev/null || true
done

echo ""
echo ">>> [Step 6] 收尾：仅导出最后一轮语料（vectors.npy + tests.jsonl）..."
mkdir -p "$FINAL_EXPORT_DIR"
$PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" \
    --dataset "$DATASETS" \
    --state-file "$DRIFT_STATE_FILE" \
    --host "$SERVER_HOST" \
    --export-final "$FINAL_EXPORT_DIR"

if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
    mkdir -p "$DRIFTING_DIR/monitoring_results"
    LOG_NAME=$(echo "$ENGINE_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')
    if [ -f "$MONITOR_DIR/docker.stats.jsonl" ]; then
        mv "$MONITOR_DIR/docker.stats.jsonl" "$DRIFTING_DIR/monitoring_results/${LOG_NAME}-drift-docker.stats-${RUN_TS}.jsonl" 2>/dev/null || true
    fi
fi

echo ""
echo "======================================="
echo "📊 完成"
echo "metrics: $DRIFT_METRICS_ARCHIVE_DIR/"
echo "live 查询+KNN（每轮覆盖）: $LIVE_EVAL_DIR/tests.jsonl"
echo "最终导出（仅此一处 vectors.npy）: $FINAL_EXPORT_DIR/"
echo "segment: $DRIFT_SEGMENT_STATS_DIR/"
echo "======================================="
