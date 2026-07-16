#!/usr/bin/env bash
set -e

# 获取脚本所在目录
SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# 默认值设置
SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
DATASETS=${3:-"random-100"} # 默认先用 random-100 跑通
SERVER_HOST="127.0.0.1"

# 定义 Milvus 目录
MILVUS_DIR="$SOURCE_DIR/engine/servers/$SERVER_PATH"
MONITOR_DIR="$SOURCE_DIR/monitoring"
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"

# Activate virtual environment if it exists
VENV_PATH=${VENV_PATH:-"/path/to/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Use Python from virtual environment if available, otherwise use python3.11
if [ -f "$VENV_PATH/bin/python3.11" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3.11"
elif [ -f "$VENV_PATH/bin/python3" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3"
else
    PYTHON_CMD="python3.11"
fi

echo "======================================="
echo "🛠️  开始测试流程"
echo "Engine: $ENGINE_NAME | Dataset: $DATASETS"
echo "======================================="

# 1. 启动 Docker 资源监控 (后台运行)
# 注意：确保 monitor_docker.sh 有执行权限
echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    # 清理旧日志
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    # 后台运行
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控进程 PID: $MONITOR_PID"
else
    echo "⚠️  未找到监控脚本，跳过监控步骤。"
fi

# 2. 重置 Milvus 环境 (Down -> Clean -> Up)
echo ">>> [Step 2] 重置 Milvus..."
cd "$MILVUS_DIR"
docker compose down -v  # 停止并删卷
sleep 5                 # 稍微缓冲一下

# 启动容器
docker compose up -d

# 3. 等待启动 (你的经验数据：90s，这里为了测试可以用短一点，比如 random-100 可能 30s 就够)
echo ">>> [Step 3] 等待服务启动 (90s)..."
sleep 90

# 4. 运行 Python 测试
echo ">>> [Step 4] 运行 Benchmark..."
# 代理设置
export no_proxy="localhost,127.0.0.1,::1"

# 切换回根目录运行脚本
cd "$SOURCE_DIR"
# 这里的 python 路径按你服务器实际情况写
$PYTHON_CMD run.py --engines "$ENGINE_NAME" --datasets "${DATASETS}" --host "$SERVER_HOST"

# 5. 测试结束，停止监控和容器
echo ">>> [Step 5] 收尾工作..."

# 杀掉监控进程
if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
    # 移动监控日志
    mkdir -p "$MONITOR_DIR/results"
    # 构造文件名
    LOG_NAME=$(echo "$ENGINE_NAME" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    mv "$MONITOR_DIR/docker.stats.jsonl" "$MONITOR_DIR/results/${LOG_NAME}-docker.stats.jsonl" 2>/dev/null || true
    echo "    监控日志已保存。"
fi

# 停止容器 (可选，如果你想保留现场查看日志，可以注释掉这行)
# cd "$MILVUS_DIR" && docker compose down

# 6. 打印结果
echo "📊 测试结果摘要:"
# 获取最新的结果文件
RES_FILE=$(ls -t results/ | grep -v 'upload' | head -n 1)
if [ -n "$RES_FILE" ]; then
    cat "results/$RES_FILE" | grep -E "mean_precisions|rps|p95_time" | sed 's/.*: \([0-9.]*\),/\1/'
else
    echo "0 0 0"
fi