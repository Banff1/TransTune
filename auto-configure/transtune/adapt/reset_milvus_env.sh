#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/../../../vector-db-benchmark-master" && pwd -P)
SERVER_PATH=${1:-milvus-single-node}
MAX_WAIT=${2:-120}

MILVUS_DIR="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH"
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"
export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"
DEFAULT_VOLUME_ROOT="${DOCKER_VOLUME_DIRECTORY}/volumes"
LEGACY_VOLUME_ROOT="$MILVUS_DIR/volumes"

DOCKER_BIN=(docker)
if ! docker info >/dev/null 2>&1; then
    if sudo -n docker info >/dev/null 2>&1; then
        DOCKER_BIN=(sudo -n docker)
        echo "    ℹ️ Docker daemon requires sudo -n, will run with sudo."
    else
        echo "    ❌ 无法访问 Docker daemon（当前用户无权限，且 sudo -n 不可用）"
        echo "       请将用户加入 docker 组，或在可 sudo 的会话中运行。"
        exit 1
    fi
fi

# 优先使用 docker compose (v2)，避免旧版 docker-compose 的 ContainerConfig 兼容性问题
if "${DOCKER_BIN[@]}" compose version >/dev/null 2>&1; then
    COMPOSE_BIN=("${DOCKER_BIN[@]}" compose)
elif command -v docker-compose >/dev/null 2>&1; then
    if [ "${DOCKER_BIN[0]}" = "sudo" ]; then
        COMPOSE_BIN=(sudo -n docker-compose)
    else
        COMPOSE_BIN=(docker-compose)
    fi
else
    echo "    ❌ 未找到 docker compose 或 docker-compose。"
    exit 1
fi

echo ">>> [Step 1] 重置 Milvus (Down -> Clean -> Up)..."
cd "$MILVUS_DIR"
"${COMPOSE_BIN[@]}" down -v || true
sleep 5

CLEAN_HOST_VOLUMES=${CLEAN_HOST_VOLUMES:-1}
if [ "$CLEAN_HOST_VOLUMES" = "1" ]; then
    VOLUME_ROOT="${VOLUME_ROOT:-$DEFAULT_VOLUME_ROOT}"
    VOLUME_ROOT_ABS="$(realpath -e "$VOLUME_ROOT" 2>/dev/null || realpath -m "$VOLUME_ROOT")"
    DEFAULT_VOLUME_ROOT_ABS="$(realpath -m "$DEFAULT_VOLUME_ROOT" 2>/dev/null || echo "$DEFAULT_VOLUME_ROOT")"
    LEGACY_VOLUME_ROOT_ABS="$(realpath -m "$LEGACY_VOLUME_ROOT" 2>/dev/null || echo "")"
    if [ -n "$VOLUME_ROOT_ABS" ] && {
        [ "$VOLUME_ROOT_ABS" = "$DEFAULT_VOLUME_ROOT_ABS" ] ||
        [[ "$VOLUME_ROOT_ABS" == */milvus-single-node/volumes ]] ||
        { [ -n "$LEGACY_VOLUME_ROOT_ABS" ] && [ "$VOLUME_ROOT_ABS" = "$LEGACY_VOLUME_ROOT_ABS" ]; };
    }; then
        echo "    🧹 清理宿主机数据目录: $VOLUME_ROOT_ABS"
        if ! rm -rf "${VOLUME_ROOT_ABS:?}/"* "${VOLUME_ROOT_ABS:?}"/.[!.]* "${VOLUME_ROOT_ABS:?}"/..?* 2>/dev/null; then
            echo "    ⚠️ 普通权限清理失败，尝试 sudo -n"
            sudo -n rm -rf "${VOLUME_ROOT_ABS:?}/"* "${VOLUME_ROOT_ABS:?}"/.[!.]* "${VOLUME_ROOT_ABS:?}"/..?* 2>/dev/null || \
                echo "    ⚠️ sudo 清理失败，继续执行（可能复用旧数据）"
        fi
    else
        echo "    ⚠️ volumes 目录不符合预期，跳过清理: $VOLUME_ROOT_ABS"
    fi
else
    echo "    ℹ️ CLEAN_HOST_VOLUMES=0，跳过清理宿主机 volumes。"
fi

"${COMPOSE_BIN[@]}" up -d

echo ">>> [Step 2] 等待 Milvus 服务就绪..."
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if ! "${DOCKER_BIN[@]}" ps | grep -q milvus-standalone; then
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
        continue
    fi

    HEALTH=$("${DOCKER_BIN[@]}" inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "    ✅ Milvus 服务已就绪"
        exit 0
    fi

    STATUS=$("${DOCKER_BIN[@]}" inspect --format='{{.State.Status}}' milvus-standalone 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "exited" ]; then
        echo "    ❌ 容器已退出，退出码: $("${DOCKER_BIN[@]}" inspect --format='{{.State.ExitCode}}' milvus-standalone 2>/dev/null || echo 'unknown')"
        "${DOCKER_BIN[@]}" logs milvus-standalone --tail 30 2>&1 || true
        exit 1
    fi

    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo "    等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)"
    fi
done

echo "    ⚠️ 等待超时（${MAX_WAIT}s），Milvus 仍未 healthy"
exit 1

