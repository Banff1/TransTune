# Dataset Sampling Benchmark

对指定数据集按比例采样 base vectors，保持 query 向量不变，分别测试原始数据集与采样后数据集的吞吐、延迟和 recall，并输出对比结果。

## 目录结构

```
sampling/
├── README.md
├── generate_sampled_dataset.py          # 生成采样数据集
├── run_sampling_benchmark.sh            # 一键采样 + benchmark + 对比
├── run_sampling_ivf_param_sweep.py      # IVF_FLAT 参数全因子 sweep
├── run_sampling_ivfpq_param_sweep.py    # IVF_PQ 参数全因子 sweep
├── run_sampling_ivfsq_param_sweep.py    # IVF-SQ 参数全因子 sweep
├── run_sampling_hnsw_param_sweep.py     # HNSW 参数全因子 sweep
├── run_sampling_scann_param_sweep.py    # SCANN 参数全因子 sweep
└── train_perfscale_model.py             # 基于多组 sampling 数据训练性能 scale 预测模型
```

依赖同目录上一级的公共脚本：

- `../generate_drift_dataset.py`：数据集解析与读取
- `../run_custom_benchmark.py`：运行 vector-db-benchmark

## 工作流程

1. 读取指定数据集（支持 `datasets.json` 中的名称、目录路径或 `.hdf5` 文件）
2. 按 `SAMPLE_RATIO` 随机采样 base vectors（保留 `payloads.jsonl`，若存在）
3. 原样复制 `queries.jsonl`
4. 基于采样后的 base vectors 重建 `neighbours.jsonl`（用于 recall 评估）
5. 分别在原始数据集和采样数据集上运行 benchmark
6. 输出性能对比摘要

## 快速开始

在仓库根目录执行：

```bash
bash auto-configure/vdtuner/new_adapt/sampling/run_sampling_benchmark.sh \
  milvus-single-node milvus-p10 random-100
```

参数说明：

| 位置参数 | 默认值 | 说明 |
|---------|--------|------|
| `SERVER_PATH` | `milvus-single-node` | Milvus docker compose 目录名 |
| `ENGINE_NAME` | `milvus-p10` | benchmark 引擎配置名 |
| `SOURCE_DATASET_PATH` | `random-100` | 源数据集（名称/路径/h5） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_RATIO` | `0.5` | 采样比例，范围 `(0, 1]` |
| `NEIGHBORS_TOP_K` | `0` | 重建 ground truth 的 K；`0` 表示从源数据集推断，否则默认 100 |
| `SEED` | `42` | 随机采样种子 |
| `DISTANCE` | 空 | 距离度量；空则使用 `datasets.json` 或 `cosine` |
| `VECTOR_SIZE` | `0` | 向量维度；`0` 表示自动检测 |
| `HOST` | `127.0.0.1` | Milvus 地址 |
| `RUN_TAG` | 自动时间戳 | 本次运行标识 |
| `VENV_PATH` | `/path/to/venv` | Python 虚拟环境路径 |

示例：采样 30% 数据

```bash
SAMPLE_RATIO=0.3 SEED=123 \
bash auto-configure/vdtuner/new_adapt/sampling/run_sampling_benchmark.sh \
  milvus-single-node milvus-p10 random-100
```

## 单独生成采样数据集

不跑 benchmark，仅生成采样数据：

```bash
python3 auto-configure/vdtuner/new_adapt/sampling/generate_sampled_dataset.py \
  --source-path random-100 \
  --output-name my-sample-run \
  --dataset-name-prefix new-adapt-my-sample-run \
  --sample-ratio 0.5
```

输出目录：

```
vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/
├── sample_info.json
└── sampled/
    ├── vectors.jsonl
    ├── queries.jsonl
    └── neighbours.jsonl
```

## 输出指标

终端和 `perf_compare_sampling_summary.json` 中包含：

| 指标 | 含义 |
|------|------|
| `rps` | 吞吐（queries/s） |
| `p95_time` / `p99_time` | P95 / P99 延迟 |
| `mean_time` | 平均单次查询延迟 |
| `mean_precisions` | 平均 recall（@K） |
| `total_time` | 总查询耗时 |

## 输出文件

每次运行在以下目录生成结果：

```
vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/
├── sample_info.json                      # 采样元信息
├── original_result_meta.json             # 原始数据集 benchmark 元信息
├── sampled_result_meta.json              # 采样数据集 benchmark 元信息
└── perf_compare_sampling_summary.json    # 性能对比摘要
```

## 参数 Sweep 脚本（`run_sampling_*_param_sweep.py`）

这组脚本用于自动化参数扫描：对每个 `(采样比例 × 索引参数)` 组合，临时修改 Milvus 引擎配置、调用 `run_sampling_benchmark.sh` 跑 benchmark，并将原始数据集 vs 采样数据集的性能指标汇总到 Excel（`.xlsx`）。

### 通用行为

- 每次运行会**临时修改** `--config-json` 中的引擎参数，全部跑完后**自动恢复**原配置。
- 每轮 benchmark 结束后会清理 `vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/`，避免磁盘占用过大。
- 每完成一轮即写入 Excel，可作为 checkpoint；中断后已完成的行会保留。
- 依赖 `openpyxl`：`pip install openpyxl`
- 建议在 `sampling/` 目录下执行，或通过绝对路径调用。

### 通用命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server-path` | `milvus-single-node` | 传给 `run_sampling_benchmark.sh` 的 server 目录 |
| `--engine-name` | `milvus-p10` | 引擎配置名（对应 config json 中的 `name`） |
| `--source-dataset` | `glove-100-angular` | 源数据集 |
| `--sampling-script` | `run_sampling_benchmark.sh` | benchmark 脚本路径（相对本文件或绝对路径） |
| `--config-json` | `vector-db-benchmark-master/experiments/configurations/milvus-single-node.json` | 引擎配置文件 |
| `--output-xlsx` | 各脚本默认值不同 | 输出 Excel 路径 |
| `--continue-on-error` | 关闭 | 某次运行失败时继续后续组合 |

### `run_sampling_ivf_param_sweep.py`（IVF_FLAT）

全因子扫描 `SAMPLE_RATIO × nlist × nprobe`，索引类型保持 config 中原有 IVF 设置。

默认 sweep 网格（可在脚本 `main()` 中修改）：

| 维度 | 取值 |
|------|------|
| `SAMPLE_RATIO` | 0.01, 0.02, 0.03, 0.05, 0.08, 0.10 |
| `nlist` | 50, 100, …, 500（步长 50） |
| `nprobe` | 10, 15, …, 50（步长 5） |

```bash
cd auto-configure/vdtuner/new_adapt/sampling
python3 run_sampling_ivf_param_sweep.py \
  --source-dataset glove-100-angular \
  --output-xlsx sampling_param_sweep_results.xlsx
```

### `run_sampling_ivfpq_param_sweep.py`（IVF_PQ）

全因子扫描 `SAMPLE_RATIO × nlist × nprobe × m × nbits`。

默认 sweep 网格：

| 维度 | 取值 |
|------|------|
| `SAMPLE_RATIO` | 0.10, 0.08, 0.05, 0.03, 0.01 |
| `nlist` | 50, 200, 350, 500 |
| `nprobe` | 10, 35 |
| `m` | 4, 20 |
| `nbits` | 4, 8 |

```bash
python3 run_sampling_ivfpq_param_sweep.py \
  --source-dataset glove-100-angular \
  --output-xlsx sampling_ivfpq_param_sweep_results.xlsx
```

### `run_sampling_ivfsq_param_sweep.py`（IVF-SQ）

全因子扫描 `SAMPLE_RATIO × nlist × nprobe`。

额外参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--index-type` | `IVF_SQ8` | Milvus IVF-SQ 索引类型，如 `IVF_SQ8H` |

默认 sweep 网格：

| 维度 | 取值 |
|------|------|
| `SAMPLE_RATIO` | 0.10, 0.08, 0.05, 0.03, 0.01 |
| `nlist` | 50, 200, 350, 500 |
| `nprobe` | 10, 20, 30, 40, 50 |

```bash
python3 run_sampling_ivfsq_param_sweep.py \
  --index-type IVF_SQ8 \
  --output-xlsx sampling_ivfsq_param_sweep_results.xlsx
```

### `run_sampling_hnsw_param_sweep.py`（HNSW）

全因子扫描 `SAMPLE_RATIO × M × efConstruction × efSearch`。

默认 sweep 网格：

| 维度 | 取值 |
|------|------|
| `SAMPLE_RATIO` | 0.01, 0.03, 0.05, 0.08, 0.10 |
| `M` | 10, 30, 50 |
| `efConstruction` | 64, 192, 320, 448 |
| `efSearch` | 101, 151, 201, 251 |

```bash
python3 run_sampling_hnsw_param_sweep.py \
  --source-dataset glove-100-angular \
  --output-xlsx sampling_hnsw_param_sweep_results.xlsx
```

### `run_sampling_scann_param_sweep.py`（SCANN）

全因子扫描 `SAMPLE_RATIO × nlist × nprobe × reorder_k`。

额外参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reuse-sampled-dataset` / `--no-reuse-sampled-dataset` | 开启 | 同一 `sample_ratio` 下复用已生成的采样数据集，减少重复采样开销 |

默认 sweep 网格：

| 维度 | 取值 |
|------|------|
| `SAMPLE_RATIO` | 0.01, 0.03, 0.05 |
| `nlist` | 100, 175, 250, 325, 400 |
| `nprobe` | 10, 60 |
| `reorder_k` | 100, 150, 200 |

```bash
python3 run_sampling_scann_param_sweep.py \
  --source-dataset glove-100-angular \
  --output-xlsx sampling_new_scann_param_sweep_results.xlsx \
  --continue-on-error
```

后台运行示例（输出重定向到日志）：

```bash
nohup python3 run_sampling_scann_param_sweep.py \
  --source-dataset glove-100-angular \
  > output_scann.log 2>&1 &
```

### Sweep 输出 Excel 列

各脚本的 Excel 均包含以下核心列（索引参数列因索引类型而异）：

| 列 | 说明 |
|----|------|
| `sample_ratio` | 采样比例 |
| `original_rps` / `sampled_rps` | 原始 / 采样数据集吞吐 |
| `original_p95_time` / `sampled_p95_time` | 原始 / 采样数据集 P95 延迟 |
| `original_mean_precisions` / `sampled_mean_precisions` | 原始 / 采样数据集 recall |
| `run_tag` | 本次运行标识 |
| `summary_json` | `perf_compare_sampling_summary.json` 路径 |
| `status` / `error` | 成功或失败信息 |

## 注意事项

- 每次 benchmark 前会重启 Milvus（`docker compose down -v && up -d`），确保索引状态一致。
- 采样后 query 不变，但 `neighbours.jsonl` 会基于采样后的 base 重新计算，recall 反映的是采样后 ground truth 下的检索质量。
- 需要已安装 Python 依赖（`numpy`, `h5py`, `scikit-learn`）以及可运行的 Milvus docker 环境。
