import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

# h5_path = "/home/user/project/VDTuner/vector-db-benchmark-master/datasets/glove-100-angular/glove-100-angular.hdf5"
# h5_path = "/home/user/project/VDTuner/vector-db-benchmark-master/datasets/deep-image-96-angular/deep-image-96-angular.hdf5"
h5_path = "/home/user/project/VDTuner/vector-db-benchmark-master/datasets/gist-960-angular/gist-960-angular.hdf5"
k = 10          # kNN 的 k
sample_size = 5000  # 为了省时间，可以只抽样一部分点计算

with h5py.File(h5_path, "r") as f:
    # ann-benchmarks 的 HDF5 一般有 'train' 和 'test'，向量都在 'train' 或 'test'
    # glove-100-angular 里通常 'train' 是基向量，'test' 是查询；这里只看 'train'
    X = f["train"][:]    # shape: (N, d)

N, d = X.shape
print("dimensionality (d):", d)
print("cardinality (N):", N)

# 抽样 (避免 N 很大时太慢)
if N > sample_size:
    idx = np.random.choice(N, size=sample_size, replace=False)
    X_sample = X[idx]
else:
    idx = np.arange(N)
    X_sample = X

# cosine 距离 = 1 - 余弦相似度；glove-100-angular 是 cosine
# 用向量归一化 + 欧式距离或直接 metric="cosine" 都可以
X_norm = X_sample / np.linalg.norm(X_sample, axis=1, keepdims=True)

nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")  # +1 是包含自己
nn.fit(X_norm)
distances, indices = nn.kneighbors(X_norm)

# distances[:, 0] 是距离自己（为0），后面 k 个是真正的 kNN
knn_distances = distances[:, 1:]               # shape: (n_sample, k)
avg_knn_dist = knn_distances.mean()
print(f"avg distance to {k}NN (cosine):", avg_knn_dist)

# 估计非 kNN 的平均距离：随机选一些非近邻点
n_sample = X_sample.shape[0]
n_non_knn_samples_per_point = k  # 每个点再采样 k 个非近邻比较

all_indices = np.arange(N)
non_knn_dists = []

for i, xi in enumerate(X_sample):
    # 近邻（在原全集中的索引）
    knn_global_idx = idx[indices[i, 1:]]  # k 个近邻在原 X 中的索引
    # 从全集中去掉近邻和自己，随机选一些当作“非近邻”
    mask = np.ones(N, dtype=bool)
    mask[knn_global_idx] = False
    mask[idx[i]] = False
    candidates = all_indices[mask]

    if len(candidates) == 0:
        continue

    pick = np.random.choice(candidates, size=min(n_non_knn_samples_per_point, len(candidates)), replace=False)
    xj = X[pick]
    # 计算与这些“非近邻”的 cosine 距离
    xi_norm = xi / np.linalg.norm(xi)
    xj_norm = xj / np.linalg.norm(xj, axis=1, keepdims=True)
    cos_sim = xj_norm @ xi_norm
    cos_dist = 1.0 - cos_sim
    non_knn_dists.extend(cos_dist.tolist())

avg_non_knn_dist = float(np.mean(non_knn_dists))
print(f"avg distance to non-{k}NN (cosine, sampled):", avg_non_knn_dist)

ratio = avg_knn_dist / avg_non_knn_dist
print(f"ratio (avg kNN distance / avg non-kNN distance):", ratio)