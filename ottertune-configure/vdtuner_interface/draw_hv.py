import re
import matplotlib.pyplot as plt
import os

# 配置路径
LOG_PATH = "pobo_record.log"
OUTPUT_IMAGE = "hypervolume_convergence.png"

def parse_log_and_get_hv(file_path):
    """
    读取日志文件，提取每一行的第一个 [RPS, Recall]，计算超体积
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return []

    hv_history = []
    
    # 正则表达式匹配行首的 [数字, 数字]
    # 格式示例: [869.5144183191475, 0.84578]
    pattern = re.compile(r"^\[([\d\.]+),\s*([\d\.]+)\]")

    print(f"开始读取 {file_path} ...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            # 提取 RPS 和 Recall
            rps = float(match.group(1))
            recall = float(match.group(2))
            
            # 计算超体积 (Hypervolume)
            # 根据 utils.py 和 optimizer 逻辑，对于单个点，RefPoint 为 [0,0]
            # Volume = (RPS - 0) * (Recall - 0)
            hv = rps * recall
            hv_history.append(hv)
            
    return hv_history

def get_monotonic_increase(data):
    """
    将数据转换为单调递增序列（历史最大值）
    """
    if not data:
        return []
        
    monotonic_data = []
    current_max = -1.0
    
    for val in data:
        if val > current_max:
            current_max = val
        monotonic_data.append(current_max)
        
    return monotonic_data

def plot_chart(hv_data):
    """
    绘制折线图
    """
    iterations = range(1, len(hv_data) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制红色虚线，带标记，模仿VDTuner论文样式
    plt.plot(iterations, hv_data, 
             color='red',           # 红色
             linestyle='--',        # 虚线
             marker='s',            # 方块标记 (square)
             markersize=4,          # 标记大小
             label='VDTuner')       # 图例
    plt.ylim(ymin=0) # 强制设置Y轴的最小值为0，最大值自动适应数据

    plt.title('Hypervolume Convergence', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    # 保存图片
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"绘图完成！图片已保存为: {OUTPUT_IMAGE}")
    # 如果有图形界面可以取消注释下面这行
    # plt.show() 

if __name__ == "__main__":
    # 1. 解析数据
    raw_hv_data = parse_log_and_get_hv(LOG_PATH)
    
    if raw_hv_data:
        # 2. 处理为单调递增（历史最优）
        monotonic_hv = get_monotonic_increase(raw_hv_data)
        
        print(f"共解析到 {len(monotonic_hv)} 条数据。")
        print(f"初始超体积: {monotonic_hv[0]:.2f}")
        print(f"最终超体积: {monotonic_hv[-1]:.2f}")
        
        # 3. 绘图
        plot_chart(monotonic_hv)
    else:
        print("日志中未提取到有效数据，请检查 pobo_record.log 格式。")