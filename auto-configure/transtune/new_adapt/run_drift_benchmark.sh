#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash auto-configure/transtune/new_adapt/run_drift_benchmark.sh [SERVER_PATH] [ENGINE_NAME] [SOURCE_DATASET_PATH]
# 示例:
#   bash auto-configure/transtune/new_adapt/run_drift_benchmark.sh milvus-single-node milvus-p10 random-100

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
PROJECT_ROOT=$(cd "$SOURCE_DIR/../../.."; pwd)
BENCHMARK_ROOT="$PROJECT_ROOT/vector-db-benchmark-master"
DATASETS_ROOT="$BENCHMARK_ROOT/datasets"
SERVERS_ROOT="$BENCHMARK_ROOT/engine/servers"

SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
SOURCE_DATASET_PATH=${3:-"random-100"}

# 漂移生成参数（可用环境变量覆盖）
N_CLUSTERS=${N_CLUSTERS:-10}
M_CLUSTERS=${M_CLUSTERS:-2}
BASE_CLUSTER_INITIAL_RATIO=${BASE_CLUSTER_INITIAL_RATIO:-0.8}
DRIFT_CLUSTER_INITIAL_RATIO=${DRIFT_CLUSTER_INITIAL_RATIO:-0.2}
KMEANS_SAMPLE=${KMEANS_SAMPLE:-50000}
KMEANS_MAX_ITER=${KMEANS_MAX_ITER:-30}
SEED=${SEED:-42}
DISTANCE=${DISTANCE:-cosine}
VECTOR_SIZE=${VECTOR_SIZE:-0}
HOST=${HOST:-127.0.0.1}
RUN_TAG=${RUN_TAG:-"drift-$(date +%Y%m%d-%H%M%S)"}

VENV_PATH=${VENV_PATH:-"/path/to/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

if [ -f "$VENV_PATH/bin/python3.12" ]; then
  PYTHON_CMD="$VENV_PATH/bin/python3.12"
elif [ -f "$VENV_PATH/bin/python3" ]; then
  PYTHON_CMD="$VENV_PATH/bin/python3"
else
  PYTHON_CMD="python3"
fi

if [ ! -d "$BENCHMARK_ROOT" ]; then
  echo "Benchmark root not found: $BENCHMARK_ROOT"
  exit 1
fi

MILVUS_DIR="$SERVERS_ROOT/$SERVER_PATH"
if [ ! -d "$MILVUS_DIR" ]; then
  echo "Milvus server directory not found: $MILVUS_DIR"
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
else
  COMPOSE_CMD="docker-compose"
fi

echo "======================================="
echo "Data drift benchmark"
echo "Engine: $ENGINE_NAME | Source: $SOURCE_DATASET_PATH | Run: $RUN_TAG"
echo "======================================="

OUTPUT_ROOT="$DATASETS_ROOT/new_adapt"
DATASET_PREFIX="new-adapt-${RUN_TAG}"

echo ">>> [1/5] Generate drift datasets..."
"$PYTHON_CMD" "$SOURCE_DIR/generate_drift_dataset.py" \
  --datasets-root "$DATASETS_ROOT" \
  --source-path "$SOURCE_DATASET_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --output-name "$RUN_TAG" \
  --dataset-name-prefix "$DATASET_PREFIX" \
  --n-clusters "$N_CLUSTERS" \
  --m-clusters "$M_CLUSTERS" \
  --base-cluster-initial-ratio "$BASE_CLUSTER_INITIAL_RATIO" \
  --drift-cluster-initial-ratio "$DRIFT_CLUSTER_INITIAL_RATIO" \
  --max-kmeans-sample "$KMEANS_SAMPLE" \
  --kmeans-max-iter "$KMEANS_MAX_ITER" \
  --seed "$SEED" \
  --distance "$DISTANCE" \
  --vector-size "$VECTOR_SIZE"

DRIFT_INFO="$OUTPUT_ROOT/$RUN_TAG/drift_info.json"
if [ ! -f "$DRIFT_INFO" ]; then
  echo "Drift info not found: $DRIFT_INFO"
  exit 1
fi

readarray -t DRIFT_META < <("$PYTHON_CMD" - "$DRIFT_INFO" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(p["dataset_name_initial"])
print(p["dataset_name_post_drift"])
print(p["initial_dir"])
print(p["post_drift_dir"])
print(p["vector_size"])
print(p["distance"])
PY
)

DATASET_INITIAL_NAME=${DRIFT_META[0]}
DATASET_POST_NAME=${DRIFT_META[1]}
DATASET_INITIAL_DIR=${DRIFT_META[2]}
DATASET_POST_DIR=${DRIFT_META[3]}
VECTOR_SIZE=${DRIFT_META[4]}
DISTANCE=${DRIFT_META[5]}

echo "Initial dataset: $DATASET_INITIAL_NAME @ $DATASET_INITIAL_DIR"
echo "Post-drift dataset: $DATASET_POST_NAME @ $DATASET_POST_DIR"

reset_milvus() {
  echo ">>> Reset Milvus ($SERVER_PATH)..." >&2
  cd "$MILVUS_DIR"
  $COMPOSE_CMD down -v || true
  sleep 3
  $COMPOSE_CMD up -d

  local max_wait=120
  local wait_count=0
  while [ $wait_count -lt $max_wait ]; do
    if ! docker ps | grep -q "milvus-standalone"; then
      echo "milvus-standalone container is not running." >&2
      docker ps >&2
      return 1
    fi
    health=$(docker inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "none")
    status=$(docker inspect --format='{{.State.Status}}' milvus-standalone 2>/dev/null || echo "unknown")
    if [ "$health" = "healthy" ] || { [ "$health" = "none" ] && [ "$status" = "running" ]; }; then
      echo "Milvus is ready (status=$status, health=$health)." >&2
      return 0
    fi
    sleep 2
    wait_count=$((wait_count + 2))
  done
  echo "Milvus wait timeout after ${max_wait}s." >&2
  return 1
}

extract_metrics() {
  "$PYTHON_CMD" - "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
res = payload["results"]
print(json.dumps({
    "rps": res.get("rps"),
    "p95_time": res.get("p95_time"),
    "p99_time": res.get("p99_time"),
    "mean_precisions": res.get("mean_precisions"),
    "mean_time": res.get("mean_time"),
    "total_time": res.get("total_time"),
}, indent=2))
PY
}

run_one() {
  local phase_name=$1
  local dataset_name=$2
  local dataset_dir=$3

  reset_milvus

  local out_json="$OUTPUT_ROOT/$RUN_TAG/${phase_name}_result_meta.json"
  echo ">>> Run phase: $phase_name" >&2
  "$PYTHON_CMD" "$SOURCE_DIR/run_custom_benchmark.py" \
    --benchmark-root "$BENCHMARK_ROOT" \
    --engine-name "$ENGINE_NAME" \
    --dataset-name "$dataset_name" \
    --dataset-path "$dataset_dir" \
    --vector-size "$VECTOR_SIZE" \
    --distance "$DISTANCE" \
    --host "$HOST" \
    --result-json "$out_json" \
    1>&2

  local result_file
  result_file=$("$PYTHON_CMD" - "$out_json" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(p["result_file"])
PY
)
  echo "$result_file"
}

echo ">>> [2/5] Run benchmark BEFORE drift (initial dataset)..."
INITIAL_RESULT_FILE=$(run_one "before_drift" "$DATASET_INITIAL_NAME" "$DATASET_INITIAL_DIR")

echo ">>> [3/5] Run benchmark AFTER drift (post-drift dataset)..."
AFTER_RESULT_FILE=$(run_one "after_drift" "$DATASET_POST_NAME" "$DATASET_POST_DIR")

echo ">>> [4/5] Summarize results..."
BEFORE_METRICS=$(extract_metrics "$INITIAL_RESULT_FILE")
AFTER_METRICS=$(extract_metrics "$AFTER_RESULT_FILE")

SUMMARY_JSON="$OUTPUT_ROOT/$RUN_TAG/perf_compare_summary.json"
"$PYTHON_CMD" - "$SUMMARY_JSON" "$DRIFT_INFO" "$INITIAL_RESULT_FILE" "$AFTER_RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
drift_info = json.load(open(sys.argv[2], "r", encoding="utf-8"))
before_result = json.load(open(sys.argv[3], "r", encoding="utf-8"))
after_result = json.load(open(sys.argv[4], "r", encoding="utf-8"))

before = before_result["results"]
after = after_result["results"]

payload = {
    "drift_info": drift_info,
    "before_drift": {
        "result_file": sys.argv[3],
        "metrics": {
            "rps": before.get("rps"),
            "p95_time": before.get("p95_time"),
            "p99_time": before.get("p99_time"),
            "mean_precisions": before.get("mean_precisions"),
            "mean_time": before.get("mean_time"),
            "total_time": before.get("total_time"),
        },
    },
    "after_drift": {
        "result_file": sys.argv[4],
        "metrics": {
            "rps": after.get("rps"),
            "p95_time": after.get("p95_time"),
            "p99_time": after.get("p99_time"),
            "mean_precisions": after.get("mean_precisions"),
            "mean_time": after.get("mean_time"),
            "total_time": after.get("total_time"),
        },
    },
}

summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(summary_path)
PY

echo ">>> [5/5] Done."
echo
echo "Before drift metrics:"
echo "$BEFORE_METRICS"
echo
echo "After drift metrics:"
echo "$AFTER_METRICS"
echo
echo "Summary saved to: $SUMMARY_JSON"
