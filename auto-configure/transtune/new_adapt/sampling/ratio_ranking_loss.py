import pandas as pd
import numpy as np

def calculate_ranking_loss(x, y):
    """
    计算两个序列之间的 Ranking Loss (逆序率)
    x: 在大数据集上的性能指标序列 (如 original_rps)
    y: 在小数据集上的性能指标序列 (如 sampled_rps)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    discordant = 0
    total_pairs = 0
    
    # 双重循环遍历所有两两组合的样本对
    for i in range(n):
        for j in range(i + 1, n):
            # 如果在 x 轴和 y 轴上的相对大小关系发生反转，则视为逆序对
            if (x[i] > x[j] and y[i] < y[j]) or (x[i] < x[j] and y[i] > y[j]):
                discordant += 1
            total_pairs += 1
            
    return discordant / total_pairs if total_pairs > 0 else 0

# 1. 读取 Excel 文件 (修改为 read_excel)
# 注意：运行前请确保已安装 openpyxl 库（pip install openpyxl）
file_path = "sampling_hnsw_param_sweep_results.xlsx" 
df = pd.read_excel(file_path)

# 2. 计算全局（Overall）的 Ranking Loss
print("================ 全局 Ranking Loss ================")
overall_rps_loss = calculate_ranking_loss(df['original_rps'], df['sampled_rps'])
overall_prec_loss = calculate_ranking_loss(df['original_mean_precisions'], df['sampled_mean_precisions'])
overall_p95_loss = calculate_ranking_loss(df['original_p95_time'], df['sampled_p95_time'])
print(f"全局吞吐量 (RPS) Ranking Loss: {overall_rps_loss:.4%}")
print(f"全局准确率 (Precision) Ranking Loss: {overall_prec_loss:.4%}")
print(f"全局 P95 延迟 (p95_time) Ranking Loss: {overall_p95_loss:.4%}\n")

# 3. 按采样率（sample_ratio）拆分计算详细排序损失
print("=========== 按采样率拆分的详细 Ranking Loss ===========")
# 使用 groupby 自动按不同采样率对数据进行切片
for ratio, group in df.groupby('sample_ratio'):
    # 获取当前采样率下的指标列
    orig_rps = group['original_rps'].values
    samp_rps = group['sampled_rps'].values
    orig_prec = group['original_mean_precisions'].values
    samp_prec = group['sampled_mean_precisions'].values
    orig_p95 = group['original_p95_time'].values
    samp_p95 = group['sampled_p95_time'].values
    
    # 分别计算当前采样率下的 RPS、Precision 和 p95_time 排序损失
    rps_loss = calculate_ranking_loss(orig_rps, samp_rps)
    prec_loss = calculate_ranking_loss(orig_prec, samp_prec)
    p95_loss = calculate_ranking_loss(orig_p95, samp_p95)
    
    print(f"当采样率 (sample_ratio) 为 {ratio} 时 (样本数: {len(group)} 行):")
    print(f"  - 吞吐量 (RPS) Ranking Loss: {rps_loss:.2%}")
    print(f"  - 准确率 (Precision) Ranking Loss: {prec_loss:.2%}")
    print(f"  - P95 延迟 (p95_time) Ranking Loss: {p95_loss:.2%}")
    print("-" * 50)