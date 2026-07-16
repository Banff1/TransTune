"""
Milvus + OtterTune-style workload characterization and task similarity.

- Prometheus metric collection during a fixed observation period
- Task profiles (aggregated metrics) for comparable workloads
- Pairwise distances as task similarity (OtterTune Sec. 4.1 analogue)
"""

from .milvus_prometheus_collector import MilvusMetricsCollector, DEFAULT_MILVUS_METRICS
from .task_similarity import (
    TaskProfile,
    aggregate_observation,
    load_profile,
    load_profiles_from_dir,
    pairwise_distance_matrix,
    save_profile,
)

__all__ = [
    "MilvusMetricsCollector",
    "DEFAULT_MILVUS_METRICS",
    "TaskProfile",
    "aggregate_observation",
    "pairwise_distance_matrix",
    "load_profiles_from_dir",
    "load_profile",
    "save_profile",
]
