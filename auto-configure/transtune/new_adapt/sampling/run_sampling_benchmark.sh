#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash auto-configure/transtune/new_adapt/sampling/run_sampling_benchmark.sh [SERVER_PATH] [ENGINE_NAME] [SOURCE_DATASET_PATH]
# 示例:
#   bash auto-configure/transtune/new_adapt/sampling/run_sampling_benchmark.sh milvus-single-node milvus-p10 random-100

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
NEW_ADAPT_DIR=$(cd "$SOURCE_DIR/.."; pwd)
PROJECT_ROOT=$(cd "$SOURCE_DIR/../../../../"; pwd)
BENCHMARK_ROOT="$PROJECT_ROOT/vector-db-benchmark-master"
DATASETS_ROOT="$BENCHMARK_ROOT/datasets"
SERVERS_ROOT="$BENCHMARK_ROOT/engine/servers"

SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
SOURCE_DATASET_PATH=${3:-"random-100"}

# compose 使用 bind mount 到宿主机目录；默认放到大盘，避免写满系统盘
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"

# 采样参数（可用环境变量覆盖）
SAMPLE_RATIO=${SAMPLE_RATIO:-0.03}
NEIGHBORS_TOP_K=${NEIGHBORS_TOP_K:-0}
SEED=${SEED:-42}
DISTANCE=${DISTANCE:-""}
VECTOR_SIZE=${VECTOR_SIZE:-0}
HOST=${HOST:-127.0.0.1}
RUN_TAG=${RUN_TAG:-"sample-$(date +%Y%m%d-%H%M%S)"}
SAMPLE_REUSE_TAG=${SAMPLE_REUSE_TAG:-"$RUN_TAG"}
REUSE_SAMPLED_DATASET=${REUSE_SAMPLED_DATASET:-1}

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

# 控制 BLAS/OpenMP 线程，避免在大并发机器上触发 OpenBLAS 线程元数据溢出或崩溃。
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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
echo "Sampling benchmark"
echo "Engine: $ENGINE_NAME | Source: $SOURCE_DATASET_PATH | Sample ratio: $SAMPLE_RATIO | Run: $RUN_TAG"
echo "Sample reuse tag: $SAMPLE_REUSE_TAG | Reuse sampled dataset: $REUSE_SAMPLED_DATASET"
echo "Threads: OPENBLAS=$OPENBLAS_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
echo "======================================="

OUTPUT_ROOT="$DATASETS_ROOT/new_adapt"
SAMPLE_OUTPUT_DIR="$OUTPUT_ROOT/$SAMPLE_REUSE_TAG"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_OUTPUT_DIR"

DATASET_PREFIX="new-adapt-${SAMPLE_REUSE_TAG}"
SAMPLE_INFO="$SAMPLE_OUTPUT_DIR/sample_info.json"
if [ "$REUSE_SAMPLED_DATASET" = "1" ] && [ -f "$SAMPLE_INFO" ]; then
  echo ">>> [1/5] Reuse sampled dataset: $SAMPLE_OUTPUT_DIR"
else
  echo ">>> [1/5] Generate sampled dataset..."
  "$PYTHON_CMD" "$SOURCE_DIR/generate_sampled_dataset.py" \
    --datasets-root "$DATASETS_ROOT" \
    --source-path "$SOURCE_DATASET_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --output-name "$SAMPLE_REUSE_TAG" \
    --dataset-name-prefix "$DATASET_PREFIX" \
    --sample-ratio "$SAMPLE_RATIO" \
    --seed "$SEED" \
    --distance "$DISTANCE" \
    --vector-size "$VECTOR_SIZE" \
    --neighbors-top-k "$NEIGHBORS_TOP_K"
fi

if [ ! -f "$SAMPLE_INFO" ]; then
  echo "Sample info not found after generation/reuse: $SAMPLE_INFO"
  exit 1
fi

readarray -t SAMPLE_META < <("$PYTHON_CMD" - "$SAMPLE_INFO" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(p["source_kind"])
print(p["source_dataset_name"])
print(p["source_dir"])
print(p["sampled_dataset_name"])
print(p["sampled_dir"])
print(p["vector_size"])
print(p["distance"])
PY
)

SOURCE_KIND=${SAMPLE_META[0]}
SOURCE_DATASET_NAME=${SAMPLE_META[1]}
SOURCE_DATASET_DIR=${SAMPLE_META[2]}
SAMPLED_DATASET_NAME=${SAMPLE_META[3]}
SAMPLED_DATASET_DIR=${SAMPLE_META[4]}
VECTOR_SIZE=${SAMPLE_META[5]}
DISTANCE=${SAMPLE_META[6]}

echo "Original dataset: $SOURCE_DATASET_NAME @ $SOURCE_DATASET_DIR"
echo "Sampled dataset: $SAMPLED_DATASET_NAME @ $SAMPLED_DATASET_DIR"

reset_milvus() {
  echo ">>> Reset Milvus ($SERVER_PATH)..." >&2
  cd "$MILVUS_DIR"
  $COMPOSE_CMD down -v || true

  # compose.yml 使用宿主机 bind mount；down -v 不会删除宿主机目录内容。
  # 若不清理，会导致 /path/to/docker 持续增长。
  local clean_host_volumes="${CLEAN_HOST_VOLUMES:-1}"
  if [ "$clean_host_volumes" = "1" ]; then
    local default_volume_root="${DOCKER_VOLUME_DIRECTORY}/volumes"
    local volume_root="${VOLUME_ROOT:-$default_volume_root}"
    local volume_root_abs
    local default_volume_root_abs
    local legacy_volume_root_abs
    volume_root_abs="$(realpath -m "$volume_root" 2>/dev/null || echo "")"
    default_volume_root_abs="$(realpath -m "$default_volume_root" 2>/dev/null || echo "$default_volume_root")"
    legacy_volume_root_abs="$(realpath -m "$MILVUS_DIR/volumes" 2>/dev/null || echo "")"

    if [ -n "$volume_root_abs" ] && {
      [ "$volume_root_abs" = "$default_volume_root_abs" ] ||
      [[ "$volume_root_abs" == */milvus-single-node/volumes ]] ||
      { [ -n "$legacy_volume_root_abs" ] && [ "$volume_root_abs" = "$legacy_volume_root_abs" ]; };
    }; then
      echo ">>> Cleanup host volume directory: $volume_root_abs" >&2
      # 用容器执行删除，避免宿主机 root 文件权限导致 rm 失败
      docker run --rm \
        -v "${volume_root_abs}:/cleanup" \
        --entrypoint /bin/sh \
        minio/minio:RELEASE.2024-12-18T13-15-44Z \
        -c 'rm -rf /cleanup/* /cleanup/.[!.]* /cleanup/..?*'
    else
      echo ">>> Skip host volume cleanup due to unexpected VOLUME_ROOT: $volume_root -> $volume_root_abs" >&2
    fi
  else
    echo ">>> CLEAN_HOST_VOLUMES=0, skip host volume cleanup." >&2
  fi

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
  local use_registry=${4:-0}

  reset_milvus

  local out_json="$RUN_OUTPUT_DIR/${phase_name}_result_meta.json"
  echo ">>> Run phase: $phase_name" >&2
  if [ "$use_registry" = "1" ]; then
    "$PYTHON_CMD" "$NEW_ADAPT_DIR/run_custom_benchmark.py" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --engine-name "$ENGINE_NAME" \
      --dataset-name "$dataset_name" \
      --host "$HOST" \
      --result-json "$out_json" \
      1>&2
  else
    "$PYTHON_CMD" "$NEW_ADAPT_DIR/run_custom_benchmark.py" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --engine-name "$ENGINE_NAME" \
      --dataset-name "$dataset_name" \
      --dataset-path "$dataset_dir" \
      --vector-size "$VECTOR_SIZE" \
      --distance "$DISTANCE" \
      --host "$HOST" \
      --result-json "$out_json" \
      1>&2
  fi

  local result_file
  result_file=$("$PYTHON_CMD" - "$out_json" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(p["result_file"])
PY
)
  if [ -z "$result_file" ] || [ ! -f "$result_file" ]; then
    echo "Result file is invalid for phase '$phase_name': '$result_file'" >&2
    return 1
  fi
  echo "$result_file"
}

echo ">>> [2/5] Run benchmark on ORIGINAL dataset..."
if [ "$SOURCE_KIND" = "h5" ]; then
  ORIGINAL_RESULT_FILE=$(run_one "original" "$SOURCE_DATASET_NAME" "$SOURCE_DATASET_DIR" 1)
else
  ORIGINAL_RESULT_FILE=$(run_one "original" "$SOURCE_DATASET_NAME" "$SOURCE_DATASET_DIR" 0)
fi

echo ">>> [3/5] Run benchmark on SAMPLED dataset..."
SAMPLED_RESULT_FILE=$(run_one "sampled" "$SAMPLED_DATASET_NAME" "$SAMPLED_DATASET_DIR" 0)

echo ">>> [4/5] Summarize results..."
ORIGINAL_METRICS=$(extract_metrics "$ORIGINAL_RESULT_FILE")
SAMPLED_METRICS=$(extract_metrics "$SAMPLED_RESULT_FILE")

SUMMARY_JSON="$RUN_OUTPUT_DIR/perf_compare_sampling_summary.json"
"$PYTHON_CMD" - "$SUMMARY_JSON" "$SAMPLE_INFO" "$ORIGINAL_RESULT_FILE" "$SAMPLED_RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
sample_info = json.load(open(sys.argv[2], "r", encoding="utf-8"))
original_result = json.load(open(sys.argv[3], "r", encoding="utf-8"))
sampled_result = json.load(open(sys.argv[4], "r", encoding="utf-8"))

o = original_result["results"]
s = sampled_result["results"]

payload = {
    "sample_info": sample_info,
    "original": {
        "result_file": sys.argv[3],
        "metrics": {
            "rps": o.get("rps"),
            "p95_time": o.get("p95_time"),
            "p99_time": o.get("p99_time"),
            "mean_precisions": o.get("mean_precisions"),
            "mean_time": o.get("mean_time"),
            "total_time": o.get("total_time"),
        },
    },
    "sampled": {
        "result_file": sys.argv[4],
        "metrics": {
            "rps": s.get("rps"),
            "p95_time": s.get("p95_time"),
            "p99_time": s.get("p99_time"),
            "mean_precisions": s.get("mean_precisions"),
            "mean_time": s.get("mean_time"),
            "total_time": s.get("total_time"),
        },
    },
}

summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(summary_path)
PY

echo ">>> [5/5] Done."
echo
echo "Original metrics:"
echo "$ORIGINAL_METRICS"
echo
echo "Sampled metrics:"
echo "$SAMPLED_METRICS"
echo
echo "Summary saved to: $SUMMARY_JSON"
