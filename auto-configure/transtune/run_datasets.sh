#!/bin/bash

# 脚本：依次执行多个数据集的自动调优任务
# 使用方法：bash run_datasets.sh
# 后台运行：nohup bash run_datasets.sh > run_datasets.log 2>&1 &

# 配置路径
WORK_DIR="/path/to/TransTune/auto-configure/transtune"
VENV_PATH="/path/to/venv/bin/activate"
MAIN_SCRIPT="main_tuner.py"
LOG_DIR="log3"
SCRIPT_LOG="run_datasets.log"

# 数据集列表（按顺序执行）
DATASETS=(
    "glove-100-angular",
    "random-match-keyword-100-angular-no-filters"
)

# 切换到工作目录
cd "$WORK_DIR" || exit 1

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 当前运行的子进程PID（用于信号处理）
CURRENT_PID=""

# 设置信号处理函数，确保子进程完成后再退出
cleanup() {
    echo "" | tee -a "$SCRIPT_LOG"
    echo "收到中断信号，等待当前任务完成..." | tee -a "$SCRIPT_LOG"
    if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        echo "等待进程 $CURRENT_PID 完成..." | tee -a "$SCRIPT_LOG"
        wait "$CURRENT_PID" 2>/dev/null
    fi
    echo "脚本已停止" | tee -a "$SCRIPT_LOG"
    exit 0
}

# 注册信号处理（但脚本本身用 nohup 运行时通常不会收到这些信号）
trap cleanup SIGTERM SIGINT SIGHUP SIGQUIT

# 记录脚本启动信息
echo "==========================================" >> "$SCRIPT_LOG"
echo "脚本启动时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SCRIPT_LOG"
echo "工作目录: $WORK_DIR" >> "$SCRIPT_LOG"
echo "==========================================" >> "$SCRIPT_LOG"

# 遍历每个数据集
for dataset in "${DATASETS[@]}"; do
    # 清理 dataset 名称：
    # - 兼容用户误写成 `"xxx",` 的情况：去掉末尾逗号
    # - 去掉首尾空白
    dataset="${dataset%,}"
    dataset="${dataset#"${dataset%%[![:space:]]*}"}"
    dataset="${dataset%"${dataset##*[![:space:]]}"}"

    echo "==========================================" | tee -a "$SCRIPT_LOG"
    echo "开始处理数据集: $dataset" | tee -a "$SCRIPT_LOG"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SCRIPT_LOG"
    echo "==========================================" | tee -a "$SCRIPT_LOG"
    
    # 修改 main_tuner.py 中的 DATASET 变量
    # 使用 sed 替换 DATASET 的值
    sed -i "s/DATASET = \".*\"/DATASET = \"$dataset\"/" "$MAIN_SCRIPT"
    
    # 验证修改是否成功
    if grep -q "DATASET = \"$dataset\"" "$MAIN_SCRIPT"; then
        echo "✓ 已更新 DATASET 为: $dataset" | tee -a "$SCRIPT_LOG"
    else
        echo "✗ 错误: 无法更新 DATASET，跳过此数据集" | tee -a "$SCRIPT_LOG"
        continue
    fi
    
    # 日志文件路径
    LOG_FILE="$LOG_DIR/${dataset}.log"
    
    echo "日志文件: $LOG_FILE" | tee -a "$SCRIPT_LOG"
    echo "启动后台任务..." | tee -a "$SCRIPT_LOG"
    
    # 激活虚拟环境并执行任务（后台运行）
    source "$VENV_PATH"
    nohup python -u "$MAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
    PID=$!
    CURRENT_PID=$PID  # 保存到全局变量供信号处理使用
    
    echo "任务已启动，进程ID: $PID" | tee -a "$SCRIPT_LOG"
    echo "等待任务完成..." | tee -a "$SCRIPT_LOG"
    
    # 等待进程完成
    # 注意：kill -0 只是检查进程是否存在，不会发送任何信号中断进程
    # 这是完全安全的，不会影响正在运行的 main_tuner.py
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60  # 每分钟检查一次
        echo "  [$(date '+%H:%M:%S')] 任务仍在运行中 (PID: $PID)..." | tee -a "$SCRIPT_LOG"
    done
    
    # 检查进程退出状态
    wait "$PID" 2>/dev/null
    EXIT_CODE=$?
    CURRENT_PID=""  # 清空，表示当前没有运行的任务
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ 进程正常退出 (退出码: $EXIT_CODE)" | tee -a "$SCRIPT_LOG"
    else
        echo "⚠ 进程退出 (退出码: $EXIT_CODE)" | tee -a "$SCRIPT_LOG"
    fi
    
    echo "✓ 数据集 $dataset 处理完成" | tee -a "$SCRIPT_LOG"
    echo "" | tee -a "$SCRIPT_LOG"
    
    # 等待一小段时间再开始下一个任务
    sleep 5
done

echo "==========================================" | tee -a "$SCRIPT_LOG"
echo "所有数据集处理完成！" | tee -a "$SCRIPT_LOG"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SCRIPT_LOG"
echo "==========================================" | tee -a "$SCRIPT_LOG"
