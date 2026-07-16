# new_adapt：数据漂移模拟与性能测试

本目录是重构后的实现，**不依赖** `auto-configure/vdtuner/adapt` 目录中的旧代码。

## 文件说明

- `generate_drift_dataset.py`
  - 读取 `vectors.jsonl`。
  - 对向量做聚类（聚成 `n` 类，KMeans）。
  - 随机选 `m` 类作为“漂移类”。
  - 构造三个数据集：
    - **初始数据集**：从 `n-m` 类按 `base_cluster_initial_ratio` 抽样 + 从 `m` 类按 `drift_cluster_initial_ratio` 抽样。
    - **drift 增量数据集**：其余剩余样本（即 `n-m` 剩余部分 + `m` 剩余部分）。
    - **drift 后数据集**：`initial + drift_increment`（用于模拟“插入 drift 数据后”的状态）。
  - 若存在 `queries.jsonl` 和 `neighbours.jsonl`，会按各子集重映射邻居索引并写入。

- `run_custom_benchmark.py`
  - 针对单个显式数据集目录运行 benchmark（无需修改 `datasets.json`）。

- `run_drift_benchmark.sh`
  - 端到端流程：
    1) 生成 `initial/drift_increment/post_drift` 数据；
    2) 重置 Milvus；
    3) 运行 drift 前测试（`initial`）；
    4) 再次重置 Milvus；
    5) 运行 drift 后测试（`post_drift`）；
    6) 汇总性能对比结果。

## 支持的数据源格式

- **jsonl 目录**：包含 `vectors.jsonl`（如 `random-100`）
- **h5 文件**：ANN Benchmarks 的 `.hdf5`（如 `glove-100-angular`，读取 `train`/`test`/`neighbors`）
- **数据集名称**：与 `datasets.json` 中的 `name` 一致时自动解析路径与 `vector_size`、`distance`

划分后会将查询向量写入 `queries.jsonl`，并按当前子集重映射 `neighbours.jsonl` 中的训练集索引。

## 快速开始

```bash
bash auto-configure/vdtuner/new_adapt/run_drift_benchmark.sh milvus-single-node milvus-p10 random-100
```

使用 GloVe（h5）示例：

```bash
bash auto-configure/vdtuner/new_adapt/run_drift_benchmark.sh milvus-single-node milvus-p10 glove-100-angular
```

## 主要参数（环境变量）

运行前可按需设置：

- `N_CLUSTERS`（默认 `10`）
- `M_CLUSTERS`（默认 `2`）
- `BASE_CLUSTER_INITIAL_RATIO`（默认 `0.8`）
- `DRIFT_CLUSTER_INITIAL_RATIO`（默认 `0.2`）
- `RUN_TAG`（默认自动时间戳）
- `VECTOR_SIZE`（默认 `0`，表示自动从首条向量推断）
- `DISTANCE`（默认 `cosine`）

示例：

```bash
N_CLUSTERS=12 M_CLUSTERS=3 BASE_CLUSTER_INITIAL_RATIO=0.8 DRIFT_CLUSTER_INITIAL_RATIO=0.2 \
RUN_TAG=exp-a bash auto-configure/vdtuner/new_adapt/run_drift_benchmark.sh milvus-single-node milvus-p10 random-100
```

## 每次运行会生成哪些新数据

`run_drift_benchmark.sh` 每执行一次，会新建一个以 `RUN_TAG` 命名的目录（默认 `drift-YYYYMMDD-HHMMSS`）。**不会修改**原始数据集（如 `glove-100-angular.hdf5`），但会把训练向量**再写一份**到 jsonl，因此大数据集时磁盘占用明显。

### 1. 漂移数据集目录（主要占用）

根目录：

`vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/`

| 路径 | 说明 |
|------|------|
| `initial/vectors.jsonl` | 漂移前的训练子集 |
| `initial/queries.jsonl` | 查询向量（从源数据导出） |
| `initial/neighbours.jsonl` | 重映射到 initial 子集内的近邻 ID |
| `drift_increment/vectors.jsonl` | 需要插入到初始库中的 drift 增量数据 |
| `drift_increment/queries.jsonl` | 查询向量（重映射到 drift_increment 子集） |
| `drift_increment/neighbours.jsonl` | 重映射到 drift_increment 子集内的近邻 ID |
| `post_drift/vectors.jsonl` | 插入 drift 后的完整数据（`initial + drift_increment`） |
| `post_drift/queries.jsonl` | 查询向量（重映射到 post_drift） |
| `post_drift/neighbours.jsonl` | 重映射到 post_drift 子集内的近邻 ID |
| `drift_info.json` | 划分参数、聚类 ID、向量条数统计等（几 KB） |
| `before_drift_result_meta.json` | 指向 drift 前 benchmark 结果文件路径 |
| `after_drift_result_meta.json` | 指向 drift 后 benchmark 结果文件路径 |
| `perf_compare_summary.json` | 前后性能指标汇总（几 KB） |

**体积估算（近似）**

- 训练向量 jsonl：约为「源训练集向量总字节数」× 文本膨胀系数（jsonl 通常比 h5 略大）。`initial` + `drift` 两条 `vectors.jsonl` 合计约等于**整份训练集再导出一次**。
- 查询相关：`queries.jsonl` + `neighbours.jsonl` 在 initial 与 drift 各一份，因此查询部分约为源数据查询文件的 **2 倍**（与训练集大小无关，通常远小于 vectors）。

以 `glove-100-angular`（约 118 万条 × 100 维）为例，一次成功运行实测约 **1.7 GB**（其中 `initial/` ~1.2G，`drift/` ~0.56G），另加查询 jsonl 各约 19M。

以 `random-100` 为例，体量很小（通常仅数 MB 级）。

### 2. Benchmark 结果（次要占用）

目录：`vector-db-benchmark-master/results/`

每次完整跑通会新增 **4 个** JSON（两次实验 × upload + search）：

- `milvus-p10-new-adapt-<RUN_TAG>-initial-upload-*.json`
- `milvus-p10-new-adapt-<RUN_TAG>-initial-search-*.json`
- `milvus-p10-new-adapt-<RUN_TAG>-post-drift-upload-*.json`
- `milvus-p10-new-adapt-<RUN_TAG>-post-drift-search-*.json`

GloVe 一次运行约 **1～2 MB** 量级（search 结果含逐查询延迟列表时更大）。引擎名不同则文件名前缀不同。

### 3. Milvus Docker 数据（可能累积）

脚本每次测试前会 `docker compose down -v` 再 `up`，但 Milvus 使用**宿主机目录 bind mount**（默认在 `$DOCKER_VOLUME_DIRECTORY/volumes/`，见 `engine/servers/milvus-single-node/docker-compose.yml`）。`down -v` **不一定**清空宿主机上的 etcd/minio/milvus 目录；多次跑大库后，该路径可能继续占磁盘。这与 `datasets/new_adapt` 无关，需单独清理。

### 4. 不会新增的内容

- 不复制原始 `.hdf5` / 源目录下的 `vectors.jsonl`
- 不在 `datasets.json` 里注册新数据集名
- 脚本目录 `auto-configure/vdtuner/new_adapt/` 本身不产生运行产物

---

## 磁盘占用与清理建议

**重复运行会线性叠加**：每多一次 `RUN_TAG`，就多一套 `datasets/new_adapt/<RUN_TAG>/` 和一组 `results/*<RUN_TAG>*` 文件。

确认不再需要某次实验后，可删除：

```bash
# 将 RUN_TAG 换成实际目录名，例如 drift-20260519-142045
RUN_TAG=drift-20260519-142045
BENCH=/path/to/TransTune/vector-db-benchmark-master

# 1) 删除该次生成的漂移数据集（释放主要空间）
rm -rf "$BENCH/datasets/new_adapt/$RUN_TAG"

# 2) 删除该次 benchmark 结果
rm -f "$BENCH/results/"*"$RUN_TAG"*

# 3) 可选：清理 Milvus 宿主机数据目录（慎用，确认路径与 compose 一致）
# sudo rm -rf /path/to/docker/volumes/{etcd,minio,milvus}/*
```

只保留汇总指标时，可只留 `perf_compare_summary.json`（或自行备份其中的 metrics），再执行上述删除。

批量清理所有漂移实验输出：

```bash
rm -rf /path/to/TransTune/vector-db-benchmark-master/datasets/new_adapt/drift-*
```

---

## 输出位置（速查）

- 漂移数据：`vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/`
- 性能汇总：`.../new_adapt/<RUN_TAG>/perf_compare_summary.json`
- Benchmark 明细：`vector-db-benchmark-master/results/`
