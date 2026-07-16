import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO

# 1. 从 CSV 文件中读取内容到字符串
file_name = "dataset_feature.csv"
try:
    with open(file_name, 'r') as f:
        csv_content = f.read()
except FileNotFoundError:
    print(f"错误：找不到文件 {file_name}。请确保文件已正确上传。")
    # 实际环境中这里应退出，但在解释器中我们会继续打印结果
    # exit()

# 2. 使用 StringIO 和 pandas 读取数据，解决 CSV 解析问题
# 移除内容中的引号，以确保正确解析
csv_content_cleaned = csv_content.replace('"', '')
df = pd.read_csv(StringIO(csv_content_cleaned))

# 3. 数据清洗和准备
performance_cols = ['Total Time', 'Mean Time', 'Mean Precisions']
feature_cols_all = ['Dimensionality (d)', 'Cardinality (N)',
                    'Avg kNN Distance', 'Avg non-kNN Distance', 'Distance Ratio']
df[feature_cols_all + performance_cols] = df[feature_cols_all + performance_cols].apply(pd.to_numeric, errors='coerce')
df['Log_Cardinality'] = np.log(df['Cardinality (N)'])
df.dropna(inplace=True)

# 4. 定义简化特征集
# 选取两个最主要的特征：维度和对数规模，避免过拟合
features_reduced = ['Dimensionality (d)', 'Log_Cardinality']
X = df[features_reduced]
X = sm.add_constant(X)
Y = df['Mean Time']

# 5. 模型拟合与验证
print("--- 特征对性能（Mean Time, 平均查询时间）的回归分析 ---")
model_reduced = sm.OLS(Y, X).fit()

# 打印回归模型的统计摘要
print(model_reduced.summary())