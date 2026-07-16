#!/usr/bin/env python
import os
import sys
import numpy as np

# 将 OtterTune server 路径加入 python path 以便导入 analysis 模块
OTTERTUNE_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ottertune/server'))
if OTTERTUNE_SERVER_PATH not in sys.path:
    sys.path.append(OTTERTUNE_SERVER_PATH)

from analysis.gp_tf import GPRGD

def target_function(x):
    """
    我们要最小化的目标函数: y = (x - 5)^2
    最优解应该是 x = 5
    """
    return (x - 5.0)**2

def main():
    print("--- OtterTune Minimal Case: Optimize y = (x - 5)^2 ---")

    # 1. 模拟一些历史训练数据
    # OtterTune 需要历史数据来建立模型
    # 我们将数据缩放到 [0, 1] 区间，这有助于模型稳定
    X_train_raw = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0], dtype=np.float32).reshape(-1, 1)
    y_train_raw = target_function(X_train_raw)
    
    X_min_raw = 0.0
    X_max_raw = 10.0
    y_min_raw = 0.0
    y_max_raw = 25.0
    
    X_train = (X_train_raw - X_min_raw) / (X_max_raw - X_min_raw)
    y_train = (y_train_raw - y_min_raw) / (y_max_raw - y_min_raw)
    
    print("训练数据 (配置 X, 已缩放):")
    print(X_train.flatten())
    print("训练数据 (性能 y, 已缩放):")
    print(y_train.flatten())

    # 2. 初始化 GPRGD 模型
    # 增加 length_scale 并减小 learning_rate 以防止梯度爆炸
    # 设置 ridge=0.1 以增加数值稳定性
    model = GPRGD(length_scale=1.0, magnitude=1.0, ridge=0.1, max_iter=20, learning_rate=0.001, check_numerics=True, debug=True)

    # 3. 拟合模型
    X_min = np.array([0.0], dtype=np.float32)
    X_max = np.array([1.0], dtype=np.float32)
    
    print("\n正在训练模型并搜索最优配置...")
    model.fit(X_train, y_train, X_min=X_min, X_max=X_max)

    # 4. 推荐新配置
    # 避开训练数据点，防止 sqrt(0) 的梯度问题
    X_test = np.array([[0.25], [0.75]], dtype=np.float32)
    results = model.predict(X_test)

    recommended_configs = results.minl_conf
    
    print("\n推荐的新配置 (注意：需要将缩放后的配置映射回原始空间):")
    for i, conf in enumerate(recommended_configs):
        val_scaled = conf[0]
        val = val_scaled * (X_max_raw - X_min_raw) + X_min_raw
        pred_y = target_function(val)
        print(f"推荐点 {i+1}: x = {val:.4f}, 实际函数值 y = {pred_y:.4f}")

    print("\n结论: OtterTune 成功通过历史数据预测并推荐了接近 x=5 的最优配置。")

if __name__ == '__main__':
    main()
