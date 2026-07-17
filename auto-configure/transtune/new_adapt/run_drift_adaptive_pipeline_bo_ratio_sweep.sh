#!/usr/bin/env bash
set -euo pipefail

# 依次运行 run_drift_adaptive_pipeline_bo.py，每次使用不同的 sample_ratio。
# 日志文件名: output_pipline_bo_scann4_ratio_<ratio>.log
#
# 用法:
#   bash run_drift_adaptive_pipeline_bo_ratio_sweep.sh
# 后台运行:
#   nohup bash run_drift_adaptive_pipeline_bo_ratio_sweep.sh > ratio_sweep_master.log 2>&1 &

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SAMPLE_RATIOS=(0.15 0.20 0.25 0.30)
LOG_PREFIX="output_pipline_bo_scann4_ratio"

VENV_PATH=${VENV_PATH:-"/path/to/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

if [ -f "$VENV_PATH/bin/python3.12" ]; then
  PYTHON_CMD="$VENV_PATH/bin/python3.12"
elif [ -f "$VENV_PATH/bin/python3" ]; then
  PYTHON_CMD="$VENV_PATH/bin/python3"
else
  PYTHON_CMD="python3"
fi

run_one() {
  local ratio=$1
  local log_file="${LOG_PREFIX}_${ratio}.log"

  echo "======================================="
  echo "sample_ratio=${ratio}"
  echo "log=${log_file}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================="

  "$PYTHON_CMD" - "$ratio" <<'PY' >"$log_file" 2>&1
import sys

import run_drift_adaptive_pipeline_bo as pipeline

ratio = float(sys.argv[1])
pipeline.CONFIG["sample_ratio"] = ratio
pipeline.main()
PY

  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "saved log: ${log_file}"
  echo
}

for ratio in "${SAMPLE_RATIOS[@]}"; do
  run_one "$ratio"
done

echo "All runs completed."
