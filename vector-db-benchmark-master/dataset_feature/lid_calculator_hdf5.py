"""
Local Intrinsic Dimension (LID) Calculator for glove-25-angular dataset
"""

import h5py
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_lid_mle(
    X: np.ndarray,
    k: int = 20,
    metric: str = "cosine",
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    use_tqdm: bool = True,
) -> tuple:
    """
    Compute Local Intrinsic Dimension (LID) using Maximum Likelihood Estimation.
    
    LID(x) = -1 / (1/k * sum(log(d_i / d_k))) for i=1 to k-1
    
    where d_i is the distance to the i-th nearest neighbor,
    and d_k is the distance to the k-th nearest neighbor.
    
    Args:
        X: Input vectors, shape (N, d)
        k: Number of nearest neighbors to use (should be >= 2)
        metric: Distance metric ('cosine', 'euclidean', 'manhattan', etc.)
        sample_size: If provided, sample this many points for computation
        random_seed: Random seed for sampling
        use_tqdm: Whether to show progress bar
    
    Returns:
        lid_values: Array of LID values for each point
        stats: Dictionary with statistics (mean, std, min, max, median)
    """
    N, d = X.shape
    
    # Sampling if needed
    if sample_size is not None and N > sample_size:
        np.random.seed(random_seed)
        idx = np.random.choice(N, size=sample_size, replace=False)
        X_sample = X[idx] if not isinstance(X, np.memmap) else X[idx]
        compute_indices = idx
    else:
        X_sample = X
        compute_indices = np.arange(N)
    
    n_compute = len(compute_indices)
    print(f"Computing LID for {n_compute} points (k={k}, metric={metric})...")
    
    # Normalize for cosine distance
    if metric == "cosine":
        X_norm = X_sample / np.linalg.norm(X_sample, axis=1, keepdims=True)
        X_norm = np.nan_to_num(X_norm)  # Handle zero vectors
    else:
        X_norm = X_sample
    
    # Compute kNN distances
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X_norm)
    distances, _ = nn.kneighbors(X_norm)
    
    # distances[:, 0] is self-distance (0), distances[:, 1:] are kNN distances
    knn_distances = distances[:, 1:]  # shape: (n_compute, k)
    
    # Compute LID for each point using MLE
    lid_values = np.zeros(n_compute)
    
    iterator = tqdm(range(n_compute), desc="Computing LID") if use_tqdm else range(n_compute)
    
    for i in iterator:
        d_k = knn_distances[i, -1]  # distance to k-th neighbor
        
        if d_k <= 0:
            lid_values[i] = np.nan
            continue
        
        # Compute sum of log(d_i / d_k) for i=1 to k-1
        d_ratios = knn_distances[i, :-1] / d_k
        d_ratios = d_ratios[d_ratios > 0]  # Remove zeros
        
        if len(d_ratios) == 0:
            lid_values[i] = np.nan
            continue
        
        log_sum = np.sum(np.log(d_ratios))
        
        if log_sum == 0:
            lid_values[i] = np.nan
        else:
            lid_values[i] = -1.0 / (log_sum / len(d_ratios))
    
    # Compute statistics
    valid_lids = lid_values[~np.isnan(lid_values)]
    
    stats = {
        "mean": float(np.mean(valid_lids)),
        "std": float(np.std(valid_lids)),
        "min": float(np.min(valid_lids)),
        "max": float(np.max(valid_lids)),
        "median": float(np.median(valid_lids)),
        "q25": float(np.percentile(valid_lids, 25)),
        "q75": float(np.percentile(valid_lids, 75)),
        "n_valid": len(valid_lids),
        "n_total": n_compute,
        "n_nan": n_compute - len(valid_lids),
    }
    
    return lid_values, stats


if __name__ == "__main__":
    # ===== 自定义参数 =====
    h5_path = "/home/user/project/VDTuner/vector-db-benchmark-master/datasets/gist-960-angular/gist-960-angular.hdf5"
    k = 10                                           # 近邻数
    metric = "cosine"                                # "cosine" / "euclidean" 等
    sample_size = 10000                              # None 表示用全部数据，10000 表示采样
    seed = 42
    # =====================
    
    # Load vectors from HDF5
    print(f"Loading vectors from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        # glove-25-angular uses 'train' key for base vectors
        if "train" in f:
            X = f["train"][:]
        elif "vectors" in f:
            X = f["vectors"][:]
        else:
            # Use first dataset
            key = list(f.keys())[0]
            X = f[key][:]
    
    N, d = X.shape
    print(f"Loaded {N} vectors of dimension {d}")
    
    # Compute LID
    lid_values, stats = compute_lid_mle(
        X,
        k=k,
        metric=metric,
        sample_size=sample_size,
        random_seed=seed,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("LID STATISTICS for glove-25-angular:")
    print(f"  Mean LID: {stats['mean']:.4f}")
    print(f"  Std LID: {stats['std']:.4f}")
    print(f"  Median LID: {stats['median']:.4f}")
    print(f"  Min LID: {stats['min']:.4f}")
    print(f"  Max LID: {stats['max']:.4f}")
    print(f"  25th percentile: {stats['q25']:.4f}")
    print(f"  75th percentile: {stats['q75']:.4f}")
    print(f"  Valid points: {stats['n_valid']}/{stats['n_total']}")
    print("=" * 60)
