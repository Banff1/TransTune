import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 自定义计算 Pairwise Ranking Loss 的函数
def calculate_pairwise_ranking_loss(y_true, y_pred):
    """
    计算成对排序损失 (Pairwise Ranking Loss)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    wrong_pairs = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                if np.sign(y_true[i] - y_true[j]) != np.sign(y_pred[i] - y_pred[j]):
                    wrong_pairs += 1
                    
    return wrong_pairs / total_pairs if total_pairs > 0 else 0

# 2. 读取 xlsx 文件数据
file_path = "sampling_param_sweep_results.xlsx"
df = pd.read_excel(file_path)

# 3. 定义特征输入 (X) 和 两个预测目标 (y_rps, y_precision)
X = df[['sample_ratio', 'nlist', 'nprobe']]

y_rps = df['original_rps'] / df['sampled_rps']
y_precision = df['original_mean_precisions'] / df['sampled_mean_precisions']

# 4. 随机划分训练集与测试集 (80% 训练集, 20% 测试集)
X_train, X_test, y_train_rps, y_test_rps = train_test_split(X, y_rps, test_size=0.2, random_state=42)
_, _, y_train_prec, y_test_prec = train_test_split(X, y_precision, test_size=0.2, random_state=42)

print(f"数据划分完成：训练集样本数 = {len(X_train)}，测试集样本数 = {len(X_test)}")

# 5. 训练模型 1 并预测：RPS 比值
model_rps = RandomForestRegressor(n_estimators=100, random_state=42)
model_rps.fit(X_train, y_train_rps)
y_pred_rps = model_rps.predict(X_test)

# 6. 训练模型 2 并预测：Precision 比值
model_precision = RandomForestRegressor(n_estimators=100, random_state=42)
model_precision.fit(X_train, y_train_prec)
y_pred_prec = model_precision.predict(X_test)

# 7. 计算指定的各项评估指标
# 模型 1 (RPS) 指标
mse_rps = mean_squared_error(y_test_rps, y_pred_rps)
r2_rps = r2_score(y_test_rps, y_pred_rps)
ranking_loss_rps = calculate_pairwise_ranking_loss(y_test_rps, y_pred_rps)

# 模型 2 (Precision) 指标
mse_prec = mean_squared_error(y_test_prec, y_pred_prec)
r2_prec = r2_score(y_test_prec, y_pred_prec)
ranking_loss_prec = calculate_pairwise_ranking_loss(y_test_prec, y_pred_prec)

print("\n================ 模型多维度指标评估 ================")
print(f"[RPS 比值预测模型]:")
print(f"  - 均方误差 (MSE): {mse_rps:.6f}      (越接近 0 越好)")
print(f"  - 决定系数 (R²):  {r2_rps:.4f}      (越接近 1 越好)")
print(f"  - 排序损失 (PRL): {ranking_loss_rps:.4f}      (越接近 0 相对顺序越准)")

print(f"\n[Precision 比值预测模型]:")
print(f"  - 均方误差 (MSE): {mse_prec:.6f}      (越接近 0 越好)")
print(f"  - 决定系数 (R²):  {r2_prec:.4f}      (越接近 1 越好)")
print(f"  - 排序损失 (PRL): {ranking_loss_prec:.4f}      (越接近 0 相对顺序越准)")
print("====================================================")

# 8. 输出所有测试集数据的详细对比表
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

comparison_df = pd.DataFrame({
    'sample_ratio': X_test['sample_ratio'].values,
    'nlist': X_test['nlist'].values,
    'nprobe': X_test['nprobe'].values,
    'Actual_RPS_Ratio': y_test_rps.values,
    'Pred_RPS_Ratio': y_pred_rps,
    'Actual_Prec_Ratio': y_test_prec.values,
    'Pred_Prec_Ratio': y_pred_prec
})

print("\n--- 所有测试集真实值与预测值对比表 ---")
print(comparison_df.to_string(index=False))