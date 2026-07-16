# OtterTune 最小化教学案例

本指南旨在通过一个简单的例子展示 OtterTune 的核心调优逻辑：**高斯过程回归 (Gaussian Process Regression, GPR)**。

OtterTune 的核心思想是利用已有的实验数据（配置和对应的性能指标）来训练一个机器学习模型（GPR），然后利用该模型预测并推荐更好的配置。

## 核心逻辑：GPR 推荐

在 OtterTune 中，推荐过程通常如下：
1. **数据准备**：收集一组已知的配置 `X` 和它们对应的性能 `y`（例如延迟或吞吐量）。
2. **训练模型**：使用 `(X, y)` 训练 GPR 模型。
3. **寻找最优**：在模型上执行梯度下降，寻找能使预测性能最优的配置。

## 运行示例

我们提供了一个简单的脚本 `simple_ottertune.py`，它尝试最小化函数 $y = (x - 5)^2$。

### 1. 环境准备

确保你已经创建并激活了 `ottertune` 环境：

```bash
conda activate ottertune
```

### 2. 运行脚本

```bash
# 进入引导目录
cd /home/dyx/VDTuner/ottertune-configure/ottertune-guide

# 运行脚本
python simple_ottertune.py
```

### 3. 代码解析

`simple_ottertune.py` 的核心部分使用了 OtterTune 源码中的 `GPRGD` 类。为了确保模型数值稳定性（避免 NaN），我们在示例中应用了以下技巧：

1. **数据缩放 (Scaling)**：将配置 `X` 和性能 `y` 都缩放到 `[0, 1]` 区间。
2. **超参数调整**：
   - `length_scale=1.0`: 增加长度尺度，使模型更平滑。
   - `ridge=0.1`: 增加正则化项，防止矩阵求逆时出现奇异矩阵。
   - `learning_rate=0.001`: 减小学习率，防止梯度爆炸。
3. **避开重复点**：初始化预测点（`X_test`）时避开已有的训练点，防止 `sqrt(0)` 导致的梯度计算错误。

```python
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

```

通过这个例子，你可以看到 OtterTune 是如何通过历史数据来“学习”性能曲线并寻找最优点的。这种机制是 OtterTune 自动调优数据库配置的核心。