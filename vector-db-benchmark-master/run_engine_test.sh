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

# compose 中 ${DOCKER_VOLUME_DIRECTORY}/volumes/{etcd,minio,milvus}；默认放到大盘，避免占满代码仓所在分区
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"

# 数据集默认使用 benchmark 仓库内 datasets/（已在 store1 大盘上）
export VDB_DATASETS_DIR="${VDB_DATASETS_DIR:-$SOURCE_DIR/datasets}"

# Activate virtual environment if it exists
VENV_PATH=${VENV_PATH:-"/path/to/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Use Python from virtual environment if available, otherwise use python3.11
if [ -f "$VENV_PATH/bin/python3.12" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3.12"
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
# 优先使用 docker compose (v2)，避免旧版 docker-compose 的 ContainerConfig 兼容性问题
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi
$COMPOSE_CMD down -v  # 停止并删卷
sleep 5                 # 稍微缓冲一下

# ⚠️ 重要：本项目的 docker-compose.yml 使用的是宿主机目录 bind mount（$DOCKER_VOLUME_DIRECTORY/volumes/...），
# `docker compose down -v` 不会删除宿主机目录里的数据。
# 如果不清理，反复跑可能导致该目录持续增长占满磁盘。
#
# 默认清理宿主机 volumes，可通过 CLEAN_HOST_VOLUMES=0 关闭。
CLEAN_HOST_VOLUMES=${CLEAN_HOST_VOLUMES:-1}
if [ "$CLEAN_HOST_VOLUMES" = "1" ]; then
    # 默认与 compose 一致：$DOCKER_VOLUME_DIRECTORY/volumes；可设 VOLUME_ROOT 覆盖
    DEFAULT_VOLUME_ROOT="${DOCKER_VOLUME_DIRECTORY}/volumes"
    VOLUME_ROOT="${VOLUME_ROOT:-$DEFAULT_VOLUME_ROOT}"
    VOLUME_ROOT_ABS="$(realpath -m "$VOLUME_ROOT" 2>/dev/null || echo "")"
    DEFAULT_VOLUME_ROOT_ABS="$(realpath -m "$DEFAULT_VOLUME_ROOT" 2>/dev/null || echo "$DEFAULT_VOLUME_ROOT")"
    LEGACY_VOLUME_ROOT="$MILVUS_DIR/volumes"
    LEGACY_VOLUME_ROOT_ABS="$(realpath -m "$LEGACY_VOLUME_ROOT" 2>/dev/null || echo "")"
    # 防呆：仅允许默认路径、旧版仓库内路径、或显式与二者之一一致
    if [ -n "$VOLUME_ROOT_ABS" ] && {
        [ "$VOLUME_ROOT_ABS" = "$DEFAULT_VOLUME_ROOT_ABS" ] ||
        [[ "$VOLUME_ROOT_ABS" == */milvus-single-node/volumes ]] ||
        { [ -n "$LEGACY_VOLUME_ROOT_ABS" ] && [ "$VOLUME_ROOT_ABS" = "$LEGACY_VOLUME_ROOT_ABS" ]; };
    }; then
        echo "    🧹 清理宿主机数据目录: $VOLUME_ROOT_ABS"
        sudo rm -rf "${VOLUME_ROOT_ABS:?}/"*
    else
        echo "    ⚠️  volumes 目录不符合预期（VOLUME_ROOT=$VOLUME_ROOT -> $VOLUME_ROOT_ABS），跳过清理。"
    fi
else
    echo "    ℹ️  CLEAN_HOST_VOLUMES=0，跳过清理宿主机 volumes。"
fi



# 启动容器
$COMPOSE_CMD up -d

#  3. 等待启动 (你的经验数据：90s，这里为了测试可以用短一点，比如 random-100 可能 30s 就够)
echo ">>> [Step 3] 等待服务启动 (90s)..."
MAX_WAIT=120
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # 检查容器是否在运行
    if ! docker ps | grep -q milvus-standalone; then
        echo "    ⚠️  milvus-standalone 容器未运行，检查日志..."
        docker logs milvus-standalone --tail 20 2>&1 | grep -E "error|fatal|panic" | head -5 || true
        echo "    ❌ 容器启动失败，退出"
        exit 1
    fi
    
    # 检查健康状态
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "    ✅ Milvus 服务已就绪"
        break
    fi
    
    # 检查容器是否退出
    STATUS=$(docker inspect --format='{{.State.Status}}' milvus-standalone 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "exited" ]; then
        echo "    ❌ 容器已退出，退出码: $(docker inspect --format='{{.State.ExitCode}}' milvus-standalone 2>/dev/null || echo 'unknown')"
        echo "    最后 30 行日志:"
        docker logs milvus-standalone --tail 30 2>&1
        exit 1
    fi
    
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo "    等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "    ⚠️  等待超时，但继续尝试运行测试..."
fi


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