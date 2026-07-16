import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# 数据集路径
dataset_path = Path("/home/user/project/VDTuner/vector-db-benchmark-master/datasets/random-100")
vectors_file = dataset_path / "vectors.jsonl"

k = 10          # kNN 的 k
sample_size = 5000  # 为了省时间，可以只抽样一部分点计算（如果数据集很大）

# 读取所有向量
print("Reading vectors from JSONL file...")
vectors = []
with open(vectors_file, "r") as f:
    for line in f:
        vector = json.loads(line.strip())
        vectors.append(vector)

X = np.array(vectors)
N, d = X.shape

print(f"dimensionality (d): {d}")
print(f"cardinality (N): {N}")

# 抽样 (避免 N 很大时太慢)
if N > sample_size:
    print(f"Sampling {sample_size} vectors from {N} total vectors...")
    idx = np.random.choice(N, size=sample_size, replace=False)
    X_sample = X[idx]
else:
    idx = np.arange(N)
    X_sample = X

# random-100 数据集使用 cosine 距离（根据 datasets.json）
# 归一化向量用于 cosine 距离计算
X_norm = X_sample / np.linalg.norm(X_sample, axis=1, keepdims=True)

print(f"Computing {k}NN distances...")
nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")  # +1 是包含自己
nn.fit(X_norm)
distances, indices = nn.kneighbors(X_norm)

# distances[:, 0] 是距离自己（为0），后面 k 个是真正的 kNN
knn_distances = distances[:, 1:]               # shape: (n_sample, k)
avg_knn_dist = knn_distances.mean()
print(f"avg distance to {k}NN (cosine): {avg_knn_dist}")

# 估计非 kNN 的平均距离：随机选一些非近邻点
print("Computing non-kNN distances...")
n_sample = X_sample.shape[0]
n_non_knn_samples_per_point = k  # 每个点再采样 k 个非近邻比较

all_indices = np.arange(N)
non_knn_dists = []

for i, xi in enumerate(X_sample):
    # 近邻（在原全集中的索引）
    knn_global_idx = idx[indices[i, 1:]]  # k 个近邻在原 X 中的索引
    # 从全集中去掉近邻和自己，随机选一些当作"非近邻"
    mask = np.ones(N, dtype=bool)
    mask[knn_global_idx] = False
    mask[idx[i]] = False
    candidates = all_indices[mask]

    if len(candidates) == 0:
        continue

    pick = np.random.choice(candidates, size=min(n_non_knn_samples_per_point, len(candidates)), replace=False)
    xj = X[pick]
    # 计算与这些"非近邻"的 cosine 距离
    xi_norm = xi / np.linalg.norm(xi)
    xj_norm = xj / np.linalg.norm(xj, axis=1, keepdims=True)
    cos_sim = xj_norm @ xi_norm
    cos_dist = 1.0 - cos_sim
    non_knn_dists.extend(cos_dist.tolist())

avg_non_knn_dist = float(np.mean(non_knn_dists))
print(f"avg distance to non-{k}NN (cosine, sampled): {avg_non_knn_dist}")

ratio = avg_knn_dist / avg_non_knn_dist
print(f"ratio (avg kNN distance / avg non-kNN distance): {ratio}")

# 输出总结
print("\n" + "="*60)
print("SUMMARY:")
print(f"  Dimensionality: {d}")
print(f"  Cardinality: {N}")
print(f"  Average {k}NN distance: {avg_knn_dist:.6f}")
print(f"  Average non-{k}NN distance: {avg_non_knn_dist:.6f}")
print(f"  Ratio (kNN / non-kNN): {ratio:.6f}")
print("="*60)

