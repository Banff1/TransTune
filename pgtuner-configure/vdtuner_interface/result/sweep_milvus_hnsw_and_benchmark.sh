#!/usr/bin/env bash
#依次修改 milvus-single-node.json 的 HNSW 参数并调用 run_engine_test.sh，
# 将每次运行的「测试结果摘要」+ 三行指标追加到结果文件。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本在 pgtuner-configure/vdtuner_interface/result/ 下，向上三级为仓库根 vdb-tuning
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCHMARK_DIR="$REPO_ROOT/vector-db-benchmark-master"
CONFIG_JSON="$BENCHMARK_DIR/experiments/configurations/milvus-single-node.json"
OUTPUT_FILE="${OUTPUT_FILE:-$SCRIPT_DIR/test_run_vdb_two_stage_pipeline_new.output}"

# 每组: [ignored, efConstruction, M, ef] —— 取第 2、3、4 个数
read -r -d '' PARAM_ROWS << 'EOF' || true
0.85 130.0 52.0 235.0
0.88 141.0 52.0 210.0
0.9 141.0 54.0 204.0
0.92 131.0 51.0 204.0
0.94 128.0 48.0 175.0
0.95 121.0 50.0 225.0
0.96 118.0 49.0 260.0
0.98 124.0 52.0 181.0
0.99 121.0 52.0 188.0
EOF

ENGINE_PATH="${1:-milvus-single-node}"
ENGINE_NAME="${2:-milvus-p10}"
DATASET="${3:-random-100-match-kw-small-vocab-no-filters}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
: >"$OUTPUT_FILE"

if [ -f "${VENV_PATH:-/path/to/venv}/bin/python3.12" ]; then
  PYTHON_CMD="${VENV_PATH:-/path/to/venv}/bin/python3.12"
elif [ -f "${VENV_PATH:-/path/to/venv}/bin/python3" ]; then
  PYTHON_CMD="${VENV_PATH:-/path/to/venv}/bin/python3"
else
  PYTHON_CMD="python3"
fi

apply_params() {
  local efc="$1" m="$2" ef="$3"
  "$PYTHON_CMD" << PY
import json
from pathlib import Path
path = Path("$CONFIG_JSON")
data = json.loads(path.read_text())
for item in data:
    if item.get("name") == "$ENGINE_NAME":
        item["search_params"][0]["params"]["ef"] = int(float("$ef"))
        item["upload_params"]["index_params"]["M"] = int(float("$m"))
        item["upload_params"]["index_params"]["efConstruction"] = int(float("$efc"))
        break
else:
    raise SystemExit("engine name not found: $ENGINE_NAME")
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
}

extract_summary() {
  "$PYTHON_CMD" - "$1" << 'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
lines = text.splitlines()
marker_idx = -1
for i, line in enumerate(lines):
    if "测试结果摘要" in line:
        marker_idx = i
if marker_idx < 0:
    print("(未找到「测试结果摘要」)", file=sys.stderr)
    sys.exit(1)
nums = []
num_re = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
for j in range(marker_idx + 1, len(lines)):
    s = lines[j].strip()
    if not s:
        continue
    if num_re.match(s):
        nums.append(s)
        if len(nums) == 3:
            break
if len(nums) != 3:
    print("(摘要后未解析到 3 个数值)", file=sys.stderr)
    sys.exit(1)
print(lines[marker_idx])
for n in nums:
    print(n)
PY
}

run_one=0
while read -r _ efc m ef; do
  [ -z "${efc:-}" ] && continue
  run_one=$((run_one + 1))
  echo ">>> [$run_one] efConstruction=$efc M=$m ef=$ef" >&2
  apply_params "$efc" "$m" "$ef"
  tmp_log="$(mktemp)"
  set +e
  (cd "$BENCHMARK_DIR" && ./run_engine_test.sh "$ENGINE_PATH" "$ENGINE_NAME" "$DATASET") >"$tmp_log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "!!! run_engine_test.sh 退出码 $rc（本组仍写入占位摘要，详见下方日志片段）" >&2
    tail -n 40 "$tmp_log" >&2
    {
      echo $'\U0001f4ca 测试结果摘要:'
      echo "0"
      echo "0"
      echo "0"
    } >>"$OUTPUT_FILE"
    echo "" >>"$OUTPUT_FILE"
    rm -f "$tmp_log"
    continue
  fi
  if ! extract_summary "$tmp_log" >>"$OUTPUT_FILE"; then
    echo "!!! 未能从日志解析测试结果摘要" >&2
    tail -n 40 "$tmp_log" >&2
    {
      echo $'\U0001f4ca 测试结果摘要:'
      echo "0"
      echo "0"
      echo "0"
    } >>"$OUTPUT_FILE"
  fi
  echo "" >>"$OUTPUT_FILE"
  rm -f "$tmp_log"
done <<<"$PARAM_ROWS"

echo ">>> 完成，结果已写入: $OUTPUT_FILE" >&2
