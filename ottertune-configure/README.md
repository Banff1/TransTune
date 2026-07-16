# OtterTune 复现与环境配置指导手册

本手册记录了在 Linux 环境下复现 OtterTune 核心算法（GPRGD）的完整过程，包括环境安装、依赖解决、以及运行中遇到的 Bug 和解决方法。

## 1. 环境准备

OtterTune 的版本较为陈旧，主要依赖 Python 3.6 和 TensorFlow 1.12.2。为了避免与现有的 PyTorch 环境冲突，**必须**创建一个独立的 Conda 环境。

### 创建 Conda 环境
```bash
conda create -n ottertune python=3.6
conda activate ottertune
```

### 安装依赖项
OtterTune 的依赖项较多，安装过程中可能会遇到网络极慢或中断的问题。

**关键经验：**
如果发现下载进度卡住（尤其是下载 TensorFlow 大包时），通常是因为系统代理干扰了国内镜像源的访问。

**解决方法：**
在执行 `pip` 前，务必先取消当前会话的代理设置：

```bash
# 进入仓库目录
cd /home/dyx/VDTuner/ottertune-configure/ottertune

# 核心：在一行命令中禁用代理并使用清华源安装
unset http_proxy https_proxy && pip install -r server/website/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装对接 VDTuner 所需的额外依赖
pip install pyyaml joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这么做？**
- `unset http_proxy https_proxy`: 确保连接不走代理，直达国内服务器。
- `-i ...`: 使用清华镜像加速。
- `&&`: 确保这两个操作在同一个执行环境下生效。

**关键依赖版本确认：**
- `tensorflow==1.12.2`
- `numpy<1.17.0` (TF 1.12 的要求)
- `scikit-learn`

---

### 安装benchmark相关的pip包

由于我们切换了一个全新的conda环境，之前安装过的benchmark相关的pip包都不见了，所以需要重新安装。否则，跑benchmark的run.py的时候会运行失败。
我在`/home/dyx/VDTuner/ottertune-configure/requirements.txt`当中列出了所有的benchmark相关的pip包，可以根据这个文件重新安装。
提示：这个文件中的pip包和之前环境的pip包可能有所不同，这是因为我们安装的`tensorflow`相关的python环境太老了，相关的依赖也会有所变化。

```txt
stopit
typer
tqdm
elasticsearch
pymilvus
opensearch-py
redis
weaviate
```

### 移除benchmark中的冗余依赖

#### 问题背景
在运行benchmark时，我们发现了两个主要问题：
1. **Qdrant依赖缺失**：代码中引用了Qdrant客户端，但我们的环境中没有安装
2. **Weaviate兼容性问题**：Weaviate客户端依赖`importlib.metadata`，而Python 3.6版本不支持该模块

由于我们的实验只需要使用Milvus向量数据库，因此可以安全地移除这些冗余的数据库依赖。

#### 解决方案
修改`/home/dyx/VDTuner/vector-db-benchmark-master/engine/clients/client_factory.py`文件，注释掉以下内容：

```python
# 注释掉Qdrant相关导入
# from engine.clients.qdrant import QdrantConfigurator, QdrantSearcher, QdrantUploader

# 注释掉Weaviate相关导入
# from engine.clients.weaviate import (
#     WeaviateConfigurator,
#     WeaviateSearcher,
#     WeaviateUploader,
# )

# 在配置字典中注释掉Qdrant和Weaviate
ENGINE_CONFIGURATORS = {
    # "qdrant": QdrantConfigurator,
    # "weaviate": WeaviateConfigurator,
    "milvus": MilvusConfigurator,
    # ... 其他保留的数据库
}

ENGINE_UPLOADERS = {
    # "qdrant": QdrantUploader,
    # "weaviate": WeaviateUploader,
    "milvus": MilvusUploader,
    # ... 其他保留的数据库
}

ENGINE_SEARCHERS = {
    # "qdrant": QdrantSearcher,
    # "weaviate": WeaviateSearcher,
    "milvus": MilvusSearcher,
    # ... 其他保留的数据库
}
```

#### 验证是否可以跑benchmark了
完成修改后，可以去到`vector-db-benchmark-master`，通过以下命令验证Milvus是否能正常运行：
```bash
python run.py --engines milvus-p10 --datasets random-100 --host 127.0.0.1
```

提示：运行上面指令的前提是启动了milvus服务，并且没有关闭它。
手动启动的方法：进入到`~/VDTuner/vector-db-benchmark-master/engine/servers/milvus-single-node$`执行过`docker compose up -d`，手动把milvus服务启动，否则会报错（跑数据集依赖于milvus服务）。

如果看到以下输出，说明修改成功：
```
(ottertune) dyx@server9050:~/VDTuner/vector-db-benchmark-master$ python run.py --engines milvus-p10 --datasets random-100 --host 127.0.0.1
Running experiment: milvus-p10 - random-100
established connection
/home/dyx/VDTuner/vector-db-benchmark-master/datasets/random-100 already exists
Experiment stage: Configure
Experiment stage: Upload
100it [00:00, 11644.70it/s]
Upload time: 1.677040726877749
Total import time: 9.320258591789752
Experiment stage: Search
10it [00:00, 3788.55it/s]
Experiment stage: Done
Results saved to:  /home/dyx/VDTuner/vector-db-benchmark-master/results
(ottertune) dyx@server9050:~/VDTuner/vector-db-benchmark-master$ 
```
注：上述的环境是ottertune，这表明，我们在一个兼容ottertune的环境中，跑通了benchmark。

## 解决QMC兼容性问题

### 问题背景
OtterTune使用的是Python 3.6和Scipy 1.0.0，而VDTuner的原始代码依赖于Scipy 1.7.0+版本中的`qmc`模块。由于Python版本差异，直接运行会出现`ImportError: cannot import name 'qmc' from 'scipy.stats'`错误。

### 解决方案
我们采用了折中方案，将VDTuner的核心接口拷贝到独立目录中进行修改：
1. 将`auto-config`中的`vdtuner`文件夹拷贝到`/home/dyx/VDTuner/ottertune-configure/vdtuner_interface`
2. 修改`vdtuner_interface/utils.py`，将`qmc`依赖替换为手动实现的拉丁超立方采样（LHS）算法

### 关键库的作用
在`main_ottertune.py`中，我们引入了三个核心库：

```python
from analysis.gp_tf import GPRGD
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule
```

#### 1. `analysis.gp_tf.GPRGD`
OtterTune的核心优化算法，基于高斯过程回归（Gaussian Process Regression）和梯度下降（Gradient Descent）的组合。它负责：
- 建立性能模型
- 预测不同配置下的性能表现
- 生成新的候选配置

#### 2. `vdtuner_interface.utils`
我们自定义的VDTuner接口，包含：
- `LHS_sample`: 手动实现的拉丁超立方采样算法，用于生成初始配置样本
- `KnobStand`: 管理数据库的可调参数（旋钮）
- `RealEnv`: 真实环境交互接口，用于执行配置和收集性能数据

#### 3. `configure`
VDTuner的配置管理模块，负责：
- `configure_index`: 配置数据库索引参数
- `filter_index_rule`: 过滤无效的索引配置
- `configure_system`: 配置系统级参数
- `filter_system_rule`: 过滤无效的系统配置

### OtterTune调优VDTuner的基本思路
1. **初始化**：使用LHS采样生成一组初始配置
2. **评估**：在真实环境中运行这些配置，收集性能数据
3. **建模**：使用GPRGD算法建立性能预测模型
4. **优化**：基于模型预测生成新的候选配置
5. **迭代**：重复评估-建模-优化过程，直到达到停止条件
6. **输出**：记录所有配置和性能数据到日志文件


### main_ottertune.py 完整代码

/home/dyx/VDTuner/ottertune-configure/main_ottertune.py
```py
#!/usr/bin/env python
import os
import sys
import numpy as np
import time
import subprocess as sp

# Add OtterTune server path to Python path
OTTERTUNE_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ottertune/server'))
if OTTERTUNE_SERVER_PATH not in sys.path:
    sys.path.append(OTTERTUNE_SERVER_PATH)

# Add VDTuner auto-configure path to Python path
VDTUNER_AUTO_CONFIGURE_PATH = os.path.abspath('/home/dyx/VDTuner/auto-configure')
if VDTUNER_AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.append(VDTUNER_AUTO_CONFIGURE_PATH)

# Import OtterTune and VDTuner modules
from analysis.gp_tf import GPRGD
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule

# Configuration
KNOB_PATH = '/home/dyx/VDTuner/auto-configure/whole_param.json'
RUN_ENGINE_PATH = '/home/dyx/VDTuner/vector-db-benchmark-master/run_engine_test.sh'

# Log file
LOG_FILE = 'ottertune_vdtuner.log'

def run_engine_test():
    """Run the engine test and return performance metrics"""
    try:
        result = sp.run(f'timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
        result = result.stdout.decode().split()
        rps = float(result[-2])
        precision = float(result[-3])
        return rps, precision
    except Exception as e:
        print(f"Error running engine test: {e}")
        return None, None

def log(message):
    """Log message to file and print"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')

def main():
    log("--- OtterTune VDTuner Integration ---\n")

    # 1. Initialize environment and knob stand
    log("Initializing environment...")
    env = RealEnv(bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH)
    knob_stand = KnobStand(KNOB_PATH)
    num_knobs = len(knob_stand.knobs_detail.keys())
    log(f"Number of knobs: {num_knobs}")

    # 2. Initial sampling using LHS
    log("\n--- Initial Sampling ---\n")
    num_initial_samples = 7
    log(f"Generating {num_initial_samples} initial samples using Latin Hypercube Sampling...")
    
    X_train = LHS_sample(num_knobs, num_initial_samples, seed=1)
    log(f"Initial samples shape: {X_train.shape}")

    # 3. Run initial samples to get performance data
    y_train = []
    for i, sample in enumerate(X_train):
        log(f"\nRunning initial sample {i+1}/{num_initial_samples}...")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], sample[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Index configuration: {index_conf}")
        log(f"System configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        rps, precision = run_engine_test()
        if rps is not None and precision is not None:
            # Single objective: maximize RPS when precision >= 0.9, else penalize
            if precision >= 0.9:
                objective = -rps  # We want to minimize, so negative RPS
            else:
                objective = 1e6  # Large penalty for low precision
            
            y_train.append(objective)
            log(f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, Objective = {objective:.4f}")
        else:
            log("Engine test failed, skipping this sample")
            X_train = np.delete(X_train, i, axis=0)
    
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    log(f"\nInitial training data shape: X={X_train.shape}, y={y_train.shape}")

    # 4. Scale performance data
    y_min = np.min(y_train)
    y_max = np.max(y_train)
    y_train_scaled = (y_train - y_min) / (y_max - y_min) if y_max > y_min else y_train

    # 5. Initialize OtterTune GPRGD model
    log("\n--- Initializing OtterTune GPRGD Model ---\n")
    model = GPRGD(
        length_scale=1.0,
        magnitude=1.0,
        ridge=0.1,
        max_iter=20,
        learning_rate=0.001,
        check_numerics=True,
        debug=True
    )

    # 6. Run iterative tuning
    log("\n--- Iterative Tuning ---\n")
    num_iterations = 200 - num_initial_samples
    
    X_min = np.zeros(num_knobs, dtype=np.float32)
    X_max = np.ones(num_knobs, dtype=np.float32)
    
    for iteration in range(num_iterations):
        log(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # Fit model to current data
        log("Fitting GPRGD model...")
        model.fit(X_train, y_train_scaled, X_min=X_min, X_max=X_max)
        
        # Generate new configuration candidates
        log("Generating new configuration candidates...")
        # Use LHS to generate candidate points
        num_candidates = 10
        X_candidates = LHS_sample(num_knobs, num_candidates, seed=iteration+2)
        
        # Predict objective for candidates
        log("Predicting objective for candidates...")
        results = model.predict(X_candidates)
        
        # Select best candidate (minimal objective)
        best_idx = np.argmin(results.minl_conf)
        best_candidate = X_candidates[best_idx]
        log(f"Best candidate found: {best_candidate}")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], best_candidate[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Best index configuration: {index_conf}")
        log(f"Best system configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        log("Running engine test with best candidate...")
        rps, precision = run_engine_test()
        
        if rps is not None and precision is not None:
            # Calculate objective
            if precision >= 0.9:
                objective = -rps
            else:
                objective = 1e6
            
            # Scale objective
            objective_scaled = (objective - y_min) / (y_max - y_min) if y_max > y_min else objective
            
            # Add to training data
            X_train = np.vstack((X_train, best_candidate.reshape(1, -1)))
            y_train = np.vstack((y_train, np.array([[objective]])))
            y_train_scaled = np.vstack((y_train_scaled, np.array([[objective_scaled]])))
            
            log(f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, Objective = {objective:.4f}")
            log(f"Updated training data shape: X={X_train.shape}, y={y_train.shape}")
        else:
            log("Engine test failed, skipping this candidate")
        
        # Update min and max for scaling
        y_min = np.min(y_train)
        y_max = np.max(y_train)

    log("\n--- Tuning Complete ---")
    log(f"Final training data shape: X={X_train.shape}, y={y_train.shape}")

if __name__ == '__main__':
    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write('')
    
    main()

```


### 验证用OtterTune调优VDTuner中的数据集

只需要在`ottertune-configure`执行`python ./main_ottertune.py`。如果看到下面的RPS = 1231.6368, Precision = 0.7671这样的字样，就说明调优成功了。
```shell
(ottertune) dyx@server9050:~/VDTuner/ottertune-configure$ python ./main_ottertune.py
[2025-12-22 11:11:16] --- OtterTune VDTuner Integration ---

[2025-12-22 11:11:16] Initializing environment...
[2025-12-22 11:11:16] Number of knobs: 16
...//这里省去很多日志
1183514it [01:56, 10143.45it/s]
10000it [00:08, 1247.92it/s]
/home/dyx/VDTuner/vector-db-benchmark-master/run_engine_test.sh: line 63: kill: (2875367) - No such process
[2025-12-22 11:16:21] Performance: RPS = 1231.6368, Precision = 0.7671, Objective = 1000000.0000
[2025-12-22 11:16:21] 
```
