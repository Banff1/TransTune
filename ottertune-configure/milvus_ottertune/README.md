# Milvus 任务画像与相似度（OtterTune 第 4 节思路）

对**不同调优任务**在**相同标准负载**下采集 Prometheus 指标，用聚合后的指标向量距离近似任务相似度，便于后续做迁移 / 先验（与论文中 workload characterization + 相似任务检索对应）。

## 0. 必须先有 Prometheus（否则会 Connection refused）

本仓库的 `run_engine_test.sh` **不会**自动启动 Prometheus。`--prometheus http://127.0.0.1:9090` 要求**本机 9090 端口上已有 Prometheus 进程**；若未启动，会出现 `Failed to establish a new connection: [Errno 111] Connection refused`。

**自检：**

```bash
curl -sS http://127.0.0.1:9090/-/ready
# 正常应返回 Prometheus is Ready.
```

**本仓库已带一套 compose（推荐）：** 在 `vector-db-benchmark-master/engine/servers/milvus-single-node/` 下已增加 `prometheus.yml`，并在 `docker-compose.yml` 里加入 `prometheus` 服务（映射宿主 `9090`）。与 Milvus 同一网络内抓取 `standalone:9091/metrics`。

```bash
cd vector-db-benchmark-master/engine/servers/milvus-single-node
docker compose up -d   # 会拉起 etcd、minio、standalone、prometheus
curl -sS http://127.0.0.1:9090/-/ready
# 在 Prometheus UI → Status → Targets 确认 milvus 为 UP
```

**自行跑 Prometheus（备选）：** 在 `scrape_configs` 里把 target 设为 Milvus 的 metrics 地址。单机 compose 已把 Milvus 指标端口映射到宿主时，可写 `127.0.0.1:9091`。

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

若 Prometheus 跑在远程主机或仅绑定在 Docker 网桥 IP，请把 `--prometheus` 改为实际可访问的 URL（例如 `http://192.168.x.x:9090`）。

---

## 依赖

```bash
pip install requests numpy typer scikit-learn
```

`scikit-learn` 仅在使用 `compare --pca N` / `--fa N` 时需要；仅用欧氏/余弦距离时可不装。

## 1. 配置 PromQL

Milvus 指标名与 `job` 标签随版本与抓取配置变化。

- **默认**：不传 `--metrics-json` 时使用 `DEFAULT_MILVUS_METRICS`（`milvus_prometheus_collector.py`），包含 **process / Go runtime / Milvus proxy·querynode·datacoord 等 30+ 项**。对 Prometheus 无结果的查询默认记为 **0**，保证 JSON 里 **metric 维度固定**，便于任务间对比；若只要“有值才写入”的旧行为，请加 **`--no-fill-missing`**。
- **自定义**：`metrics_prometheus.example.json` 可复制为 `metrics_prometheus.json` 后按需增删。

在 Prometheus UI 中先验证关键查询有非空结果。

## 2. 采集单个任务的画像

脚本位于 `milvus_ottertune/run_task_characterization.py`，请在 **`ottertune-configure` 目录**下用 **模块方式**运行（保证 `import milvus_ottertune` 正确）：

**方式 A：与负载并行采集（推荐）**  
对**每个任务**使用**同一条** `--workload-cmd`（标准负载），仅改数据集名等；负载结束即停止轮询（避免采到空闲状态）。

```bash
cd /path/to/ottertune-configure

python -m milvus_ottertune.run_task_characterization characterize \
  --prometheus http://127.0.0.1:9090 \
  --task-id glove-100-angular \
  --workload-cmd "timeout 900 ./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular" \
  --workload-cwd /path/to/vector-db-benchmark-master \
  --profiles-dir ./milvus_ottertune/task_profiles \
  --samples 200 \
  --interval 10
```

**方式 B：仅定时拉取**（负载由你手动或其它脚本启动）

```bash
python -m milvus_ottertune.run_task_characterization characterize \
  --prometheus http://127.0.0.1:9090 \
  --task-id manual-run-1 \
  --samples 30 \
  --interval 10 \
  --profiles-dir ./milvus_ottertune/task_profiles
```

输出：`task_profiles/<task_id>.json`，内含各指标在时间窗上的 **mean / std**。

## 3. 任务间相似度（距离矩阵）

对同一目录下多个 `*.json` 画像：

```bash
python -m milvus_ottertune.run_task_characterization compare \
  --profiles-dir ./milvus_ottertune/task_profiles \
  --metric euclidean \
  --out-json ./milvus_ottertune/similarity_matrix.json
```

- 默认会对**跨任务的特征**做列方向 z-score，再算距离（避免量纲主导）。
- `--metric cosine`：余弦距离 `1 - cos_sim`。
- `--pca 5`：先用 PCA 降维再算距离（需 sklearn）。
- `--fa 5`：用 **FactorAnalysis**（与论文中因子分析更贴近）；若指定则优先于 `--pca`。
- `--no-standardize` / `--no-std-features`：关闭标准化或去掉 std 特征向量。

## 4. 与 `main_ottertune.py` 的关系

当前 `main_ottertune.py` 仍是**离线 GPRGD**；本模块提供**任务侧画像与相似度**，可后续接：按距离选历史任务、对 GPR 先验或初始 LHS 做加权。需要时可再把这些 JSON 读入调优主循环。
