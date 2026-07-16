"""
Prometheus-based Milvus metrics collection for OtterTune-style workload characterization.

Metric names / labels depend on your Milvus and Prometheus scrape config; override via
MilvusMetricsCollector(metrics_map=...) or a JSON file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger(__name__)

# Default PromQL snippets aligned with OtterTune-style numeric workload features (Sec. 4.1).
# - Use {job='milvus'} when your prometheus.yml sets job_name: milvus (recommended).
# - Some milvus_* names differ by Milvus version; unknown series return empty → filled with 0 if fill_missing=True.
# - Go/process metrics usually exist on the same Milvus /metrics scrape.
DEFAULT_MILVUS_METRICS: Dict[str, str] = {
    # --- CPU / memory / process (client_golang) ---
    "cpu_usage_percent": "avg(rate(process_cpu_seconds_total{job='milvus'}[1m])) * 100",
    "memory_usage_bytes": "sum(process_resident_memory_bytes{job='milvus'})",
    "process_virtual_memory_bytes": "sum(process_virtual_memory_bytes{job='milvus'})",
    "process_open_fds": "sum(process_open_fds{job='milvus'})",
    "process_max_fds": "max(process_max_fds{job='milvus'})",
    "process_start_time_seconds": "max(process_start_time_seconds{job='milvus'})",
    # --- Go runtime ---
    "go_goroutines": "sum(go_goroutines{job='milvus'})",
    "go_threads": "sum(go_threads{job='milvus'})",
    "go_memstats_heap_inuse_bytes": "sum(go_memstats_heap_inuse_bytes{job='milvus'})",
    "go_memstats_heap_alloc_bytes": "sum(go_memstats_heap_alloc_bytes{job='milvus'})",
    "go_memstats_stack_inuse_bytes": "sum(go_memstats_stack_inuse_bytes{job='milvus'})",
    "go_memstats_sys_bytes": "sum(go_memstats_sys_bytes{job='milvus'})",
    "go_memstats_alloc_bytes": "sum(go_memstats_alloc_bytes{job='milvus'})",
    "go_memstats_mallocs_total_rate": "sum(rate(go_memstats_mallocs_total{job='milvus'}[1m]))",
    "go_memstats_frees_total_rate": "sum(rate(go_memstats_frees_total{job='milvus'}[1m]))",
    "go_gc_duration_seconds_rate": "sum(rate(go_gc_duration_seconds_sum{job='milvus'}[1m]))",
    # --- Proxy / search / insert (counters; sum without rate for “stock” view) ---
    "proxy_search_vectors_total": "sum(milvus_proxy_search_vectors_count_total{job='milvus'})",
    "proxy_insert_vectors_total": "sum(milvus_proxy_insert_vectors_count_total{job='milvus'})",
    "proxy_search_req_count_total": "sum(milvus_proxy_search_req_count{job='milvus'})",
    "proxy_insert_req_count_total": "sum(milvus_proxy_insert_req_count{job='milvus'})",
    "proxy_delete_req_count_total": "sum(milvus_proxy_delete_req_count{job='milvus'})",
    "proxy_upsert_req_count_total": "sum(milvus_proxy_upsert_req_count{job='milvus'})",
    "proxy_req_count_rate": "sum(rate(milvus_proxy_req_count{job='milvus'}[1m]))",
    "proxy_req_size_bytes_rate": "sum(rate(milvus_proxy_req_size_bytes_total{job='milvus'}[1m]))",
    "proxy_search_latency_p99_ms": (
        "histogram_quantile(0.99, sum(rate(milvus_proxy_search_latency_ms_bucket{job='milvus'}[1m])) by (le))"
    ),
    "proxy_search_latency_avg_ms": "avg(milvus_proxy_search_latency_ms{job='milvus'})",
    # --- Query node / data coord / root coord (often present in standalone) ---
    "querynode_cache_hit_ratio": "avg(milvus_querynode_cache_hit_ratio{job='milvus'})",
    "querynode_search_latency_p99_ms": (
        "histogram_quantile(0.99, sum(rate(milvus_querynode_search_latency_ms_bucket{job='milvus'}[1m])) by (le))"
    ),
    "datacoord_segment_count": "sum(milvus_datacoord_segment_count{job='milvus'})",
    "rootcoord_index_build_latency_ms": "avg(milvus_rootcoord_index_build_latency_ms{job='milvus'})",
    # --- Storage I/O ---
    "storage_download_bytes_rate": "sum(rate(milvus_storage_download_size_bytes_total{job='milvus'}[1m]))",
    "storage_upload_bytes_rate": "sum(rate(milvus_storage_upload_size_bytes_total{job='milvus'}[1m]))",
}


class MilvusMetricsCollector:
    """Fetch instant-vector values from Prometheus HTTP API."""

    def __init__(
        self,
        prometheus_url: str,
        metrics_map: Optional[Dict[str, str]] = None,
        timeout_sec: float = 30.0,
        verify_tls: bool = True,
    ):
        """
        :param prometheus_url: Base URL, e.g. http://localhost:9090
        :param metrics_map: name -> PromQL instant query
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.query_api = f"{self.prometheus_url}/api/v1/query"
        self.metrics_map = metrics_map if metrics_map is not None else dict(DEFAULT_MILVUS_METRICS)
        self.timeout_sec = timeout_sec
        self.verify_tls = verify_tls

    @classmethod
    def from_json(cls, prometheus_url: str, path: str | Path, **kwargs) -> "MilvusMetricsCollector":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics") or data
        if not isinstance(metrics, dict):
            raise ValueError("JSON must be an object mapping metric_name -> promql")
        # Drop comment / non-string values
        metrics = {k: v for k, v in metrics.items() if isinstance(v, str) and not k.startswith("_")}
        return cls(prometheus_url, metrics_map=metrics, **kwargs)

    def fetch_metric(self, prom_ql: str) -> Optional[float]:
        """Execute instant query; return scalar value or None on empty/error."""
        try:
            r = requests.get(
                self.query_api,
                params={"query": prom_ql},
                timeout=self.timeout_sec,
                verify=self.verify_tls,
            )
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") != "success":
                LOG.warning("Prometheus non-success: %s", payload)
                return None
            results = (payload.get("data") or {}).get("result") or []
            if not results:
                return None
            val = results[0].get("value")
            if not val or len(val) < 2:
                return None
            return float(val[1])
        except Exception as e:
            LOG.warning("Error fetching %s: %s", prom_ql, e)
            return None

    def collect_all_metrics_with_fetch_status(
        self,
        *,
        fill_missing: bool = True,
        missing_value: float = 0.0,
    ) -> tuple[Dict[str, float], Dict[str, bool]]:
        """
        Returns (values, fetch_ok) where fetch_ok[name] is True iff Prometheus returned a scalar
        for that query (False when empty/error and value was filled or omitted).
        """
        current: Dict[str, float] = {}
        fetch_ok: Dict[str, bool] = {}
        for name, ql in self.metrics_map.items():
            v = self.fetch_metric(ql)
            if v is not None:
                current[name] = v
                fetch_ok[name] = True
            else:
                fetch_ok[name] = False
                if fill_missing:
                    current[name] = missing_value
                    LOG.debug("Metric %s empty; using fill value %s", name, missing_value)
        return current, fetch_ok

    def collect_all_metrics(
        self,
        *,
        fill_missing: bool = True,
        missing_value: float = 0.0,
    ) -> Dict[str, float]:
        """
        One snapshot over all configured metric names.

        If fill_missing is True, any PromQL that returns no series gets missing_value so that
        task profiles keep a fixed feature dimension across runs (recommended for similarity).
        """
        d, _ = self.collect_all_metrics_with_fetch_status(
            fill_missing=fill_missing, missing_value=missing_value
        )
        return d

    def observe_period(
        self,
        num_samples: int,
        interval_sec: float,
        on_sample: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Observation period [cite OtterTune: repeated samples over workload window].

        :param num_samples: number of polling rounds
        :param interval_sec: sleep between rounds
        :param on_sample: optional callback(dict_snapshot) after each sample
        :return: list of rows, each row includes metric keys + timestamp_epoch
        """
        history: List[Dict[str, Any]] = []
        for i in range(num_samples):
            snap = self.collect_all_metrics()
            snap["timestamp_epoch"] = time.time()
            snap["sample_index"] = i
            history.append(snap)
            if on_sample:
                on_sample(snap)
            if i < num_samples - 1 and interval_sec > 0:
                time.sleep(interval_sec)
        return history
