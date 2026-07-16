import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import numpy as np

# Load the dataset
file_name = "data_feature.csv"
df = pd.read_csv(file_name)

# 定义特征和标签列
feature_cols = [
    'Dimensionality (d)', 'Cardinality (N)', 'Avg kNN Distance',
    'Avg non-kNN Distance', 'Distance Ratio', 'Mean LID'
]
label_cols = [
    'Total Time', 'Mean Time', 'Mean Precisions'
]

X = df[feature_cols]
Y = df[label_cols]

# --- 分析 1: 特征与标签的皮尔逊相关性矩阵 ---
data_for_corr = pd.concat([X, Y], axis=1)
full_corr_matrix = data_for_corr.corr()
correlation_matrix = full_corr_matrix.loc[feature_cols, label_cols]

# --- 分析 2: 随机森林回归模型的拟合优度 (R^2) ---
# 由于样本量极小，模型拟合在整个数据集上，R^2 仅代表特征的潜在映射能力。
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, Y)
Y_pred = model.predict(X)

r2_scores = {}
for i, col in enumerate(label_cols):
    r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
    r2_scores[col] = r2

# 输出结果...
print("--- Pearson Correlation Matrix (Features vs. Labels) ---")
# Use to_markdown for clean output table
print(correlation_matrix.to_markdown(floatfmt=".4f"))

print(f"\n--- Random Forest Regression R^2 on Training Data (N={len(df)}) ---")
print("NOTE: These R^2 values are likely highly OVERFITTED due to the small sample size.")
print("R^2 interpretation:")
print("  - R^2 = 1.0: Perfect prediction (model explains 100% of variance)")
print("  - R^2 = 0.8: Model explains 80% of variance")
print("  - R^2 = 0.0: Model performs no better than predicting the mean")
print("  - R^2 < 0.0: Model performs worse than predicting the mean")
print()
for col, r2 in r2_scores.items():
    print(f"R^2 for {col}: {r2:.4f}")

# --- 分析 3: 留一法交叉验证 (Leave-One-Out Cross-Validation) ---
print(f"\n--- Leave-One-Out Cross-Validation (LOOCV) Results (N={len(df)}) ---")
print("LOOCV: For each sample, train on all other samples and test on the left-out sample.")
print("This provides a more realistic assessment of model generalization ability.\n")

loo = LeaveOneOut()
X_array = X.values
Y_array = Y.values

# 存储每个标签的交叉验证结果
cv_results = {}

for label_idx, label_name in enumerate(label_cols):
    y_true_list = []
    y_pred_list = []
    
    # 对每个样本进行留一法验证
    for train_idx, test_idx in loo.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train = Y_array[train_idx, label_idx]
        y_test = Y_array[test_idx, label_idx]
        
        # 训练模型
        model_cv = RandomForestRegressor(n_estimators=10, random_state=42)
        model_cv.fit(X_train, y_train)
        
        # 预测
        y_pred = model_cv.predict(X_test)
        
        y_true_list.append(y_test[0])
        y_pred_list.append(y_pred[0])
    
    # 计算评估指标
    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    
    r2_cv = r2_score(y_true_array, y_pred_array)
    mae_cv = mean_absolute_error(y_true_array, y_pred_array)
    rmse_cv = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
    
    cv_results[label_name] = {
        'R²': r2_cv,
        'MAE': mae_cv,
        'RMSE': rmse_cv
    }
    
    print(f"--- {label_name} ---")
    print(f"  R² Score:  {r2_cv:.4f}")
    print(f"  MAE:       {mae_cv:.4f}")
    print(f"  RMSE:      {rmse_cv:.4f}")
    print()

# 总结对比
print("--- Summary: Training R² vs LOOCV R² ---")
print("(Training R² may be inflated due to overfitting, LOOCV R² is more reliable)")
print()
for col in label_cols:
    train_r2 = r2_scores[col]
    cv_r2 = cv_results[col]['R²']
    print(f"{col}:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  LOOCV R²:    {cv_r2:.4f}")
    print(f"  Difference:  {train_r2 - cv_r2:.4f}")
    print()

