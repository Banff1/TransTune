#!/usr/bin/env python3
"""Compute ranking loss using every 3rd row from the sweep results xlsx."""

import pandas as pd
import numpy as np

ROW_STRIDE = 3
file_path = "sampling_scann_param_sweep_results.xlsx"


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

    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] > x[j] and y[i] < y[j]) or (x[i] < x[j] and y[i] > y[j]):
                discordant += 1
            total_pairs += 1

    return discordant / total_pairs if total_pairs > 0 else 0


def subsample_every_n(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    return df.iloc[::stride].copy()


# 1. 读取 Excel 文件
# 注意：运行前请确保已安装 openpyxl 库（pip install openpyxl）
df = pd.read_excel(file_path, engine="openpyxl")

if "status" in df.columns:
    df = df[df["status"] == "ok"]

metric_cols = [
    "sample_ratio",
    "original_rps",
    "sampled_rps",
    "original_mean_precisions",
    "sampled_mean_precisions",
    "original_p95_time",
    "sampled_p95_time",
]
missing = [col for col in metric_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=metric_cols).copy()
df_sub = subsample_every_n(df, ROW_STRIDE)

print(f"xlsx: {file_path}")
print(f"row_stride: every {ROW_STRIDE} rows, keep 1 row")
print(f"rows before subsample: {len(df)}")
print(f"rows after subsample: {len(df_sub)}")
print()

# 2. 计算全局（Overall）的 Ranking Loss
print("================ 全局 Ranking Loss ================")
overall_rps_loss = calculate_ranking_loss(df_sub["original_rps"], df_sub["sampled_rps"])
overall_prec_loss = calculate_ranking_loss(
    df_sub["original_mean_precisions"], df_sub["sampled_mean_precisions"]
)
overall_p95_loss = calculate_ranking_loss(
    df_sub["original_p95_time"], df_sub["sampled_p95_time"]
)
print(f"全局吞吐量 (RPS) Ranking Loss: {overall_rps_loss:.4%}")
print(f"全局准确率 (Precision) Ranking Loss: {overall_prec_loss:.4%}")
print(f"全局 P95 延迟 (p95_time) Ranking Loss: {overall_p95_loss:.4%}\n")

# 3. 按采样率（sample_ratio）拆分计算详细排序损失
print("=========== 按采样率拆分的详细 Ranking Loss ===========")
for ratio, group in df.groupby("sample_ratio", sort=True):
    group_sub = subsample_every_n(group, ROW_STRIDE)

    orig_rps = group_sub["original_rps"].values
    samp_rps = group_sub["sampled_rps"].values
    orig_prec = group_sub["original_mean_precisions"].values
    samp_prec = group_sub["sampled_mean_precisions"].values
    orig_p95 = group_sub["original_p95_time"].values
    samp_p95 = group_sub["sampled_p95_time"].values

    rps_loss = calculate_ranking_loss(orig_rps, samp_rps)
    prec_loss = calculate_ranking_loss(orig_prec, samp_prec)
    p95_loss = calculate_ranking_loss(orig_p95, samp_p95)

    print(
        f"当采样率 (sample_ratio) 为 {ratio} 时 "
        f"(原始样本数: {len(group)} 行, 抽样后: {len(group_sub)} 行):"
    )
    print(f"  - 吞吐量 (RPS) Ranking Loss: {rps_loss:.2%}")
    print(f"  - 准确率 (Precision) Ranking Loss: {prec_loss:.2%}")
    print(f"  - P95 延迟 (p95_time) Ranking Loss: {p95_loss:.2%}")
    print("-" * 50)
