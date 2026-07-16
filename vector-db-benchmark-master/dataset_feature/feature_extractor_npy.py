import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def main():
    # Dataset paths
    vectors_path = Path(
        "/home/user/project/VDTuner/vector-db-benchmark-master/datasets/arxiv-titles-384-angular/arxiv_no_filters/vectors.npy"
    )

    # Parameters
    k = 10
    sample_size = 10000  # adjust if you want more/less sampling

    # Load vectors with memmap to avoid high memory usage
    print(f"Loading vectors (memmap) from: {vectors_path}")
    X = np.load(vectors_path, mmap_mode="r")
    N, d = X.shape

    print(f"dimensionality (d): {d}")
    print(f"cardinality (N): {N}")

    # Sampling to speed up computation on 1M vectors
    if N > sample_size:
        print(f"Sampling {sample_size} vectors from {N} total vectors...")
        idx = np.random.choice(N, size=sample_size, replace=False)
        X_sample = X[idx]
    else:
        idx = np.arange(N)
        X_sample = X

    # Cosine distance -> normalize then use cosine metric
    print(f"Computing {k}NN distances...")
    X_norm = X_sample / np.linalg.norm(X_sample, axis=1, keepdims=True)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")  # +1 includes self
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    # distances[:, 0] is self-distance (0); the rest are true kNNs
    knn_distances = distances[:, 1:]
    avg_knn_dist = float(knn_distances.mean())
    print(f"avg distance to {k}NN (cosine): {avg_knn_dist}")

    # Estimate non-kNN average distance by random sampling
    print("Computing non-kNN distances (sampled)...")
    n_sample = X_sample.shape[0]
    n_non_knn_samples_per_point = k  # sample k non-neighbors per point

    all_indices = np.arange(N)
    non_knn_dists = []

    for i, xi in enumerate(X_sample):
        knn_global_idx = idx[indices[i, 1:]]  # k neighbors in full X

        mask = np.ones(N, dtype=bool)
        mask[knn_global_idx] = False
        mask[idx[i]] = False
        candidates = all_indices[mask]
        if len(candidates) == 0:
            continue

        pick = np.random.choice(
            candidates,
            size=min(n_non_knn_samples_per_point, len(candidates)),
            replace=False,
        )
        xj = X[pick]

        xi_norm = xi / np.linalg.norm(xi)
        xj_norm = xj / np.linalg.norm(xj, axis=1, keepdims=True)
        cos_sim = xj_norm @ xi_norm
        cos_dist = 1.0 - cos_sim
        non_knn_dists.extend(cos_dist.tolist())

    avg_non_knn_dist = float(np.mean(non_knn_dists))
    ratio = avg_knn_dist / avg_non_knn_dist

    print(f"avg distance to non-{k}NN (cosine, sampled): {avg_non_knn_dist}")
    print(f"ratio (avg kNN distance / avg non-kNN distance): {ratio}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Dimensionality: {d}")
    print(f"  Cardinality: {N}")
    print(f"  Average {k}NN distance: {avg_knn_dist:.6f}")
    print(f"  Average non-{k}NN distance: {avg_non_knn_dist:.6f}")
    print(f"  Ratio (kNN / non-kNN): {ratio:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

