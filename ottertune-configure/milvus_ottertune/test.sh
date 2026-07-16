#!/bin/bash

# 1. 基础路径配置
VENV_PATH="/path/to/venv/bin/activate"
MILVUS_DIR="/path/to/TransTune/vector-db-benchmark-master/engine/servers/milvus-single-node"
WORKLOAD_CWD="/path/to/TransTune/vector-db-benchmark-master"
BASE_DIR=$(pwd)

# 激活虚拟环境
source "$VENV_PATH"

# 定义数据集列表
datasets=(
    "glove-25-angular"
    "deep-image-96-angular"
    "random-geo-radius-2048-angular-no-filters"
    "random-100-match-kw-small-vocab-no-filters"
    "random-match-int-100-angular-no-filters"
    "random-range-2048-angular-no-filters"
)

# 2. 循环执行任务
for ds in "${datasets[@]}"
do
    echo "======================================================"
    echo "开始处理数据集: $ds"
    echo "======================================================"

    # --- 步骤 A: 启动环境 ---
    echo "[Step 1] 启动 Milvus 容器组..."
    cd "$MILVUS_DIR" || exit
    docker compose up -d

    # 等待 Prometheus 就绪 (使用 curl 轮询接口)
    echo "[Step 2] 等待 Prometheus 服务就绪..."
    until curl -sS http://127.0.0.1:9090/-/ready > /dev/null 2>&1; do
        printf "."
        sleep 2
    done
    echo -e "\n服务已就绪！"

    # --- 步骤 B: 执行任务 ---
    echo "[Step 3] 执行特征刻画任务..."
    cd "$BASE_DIR" || exit
    python run_task_characterization.py characterize \
      --prometheus http://127.0.0.1:9090 \
      --task-id "$ds" \
      --workload-cmd "timeout 900 ./run_engine_test.sh milvus-single-node milvus-p10 $ds" \
      --workload-cwd "$WORKLOAD_CWD" \
      --profiles-dir ./task_profiles \
      --samples 30 --interval 10

    # --- 步骤 C: 环境清理 (按要求先 down 再 rm) ---
    echo "[Step 4] 停止容器并清理数据卷..."
    cd "$MILVUS_DIR" || exit
    docker compose down
    
    echo "[Step 5] 物理删除数据目录..."
    sudo rm -rf volumes/*
    
    echo ">>> 数据集 $ds 处理完成。"
    echo "------------------------------------------------------"
    cd "$BASE_DIR" || exit
done

echo "所有数据集的基准测试已全部执行完毕！"