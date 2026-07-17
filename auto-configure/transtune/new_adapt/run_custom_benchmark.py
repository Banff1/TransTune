#!/usr/bin/env python3
"""Run vector-db-benchmark for an explicit jsonl dataset path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a custom benchmark dataset.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Path to dataset folder; absolute or relative to benchmark datasets root.",
    )
    parser.add_argument("--vector-size", type=int, default=0)
    parser.add_argument("--distance", default="cosine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild Milvus vector index before search. Useful with --skip-upload.",
    )
    parser.add_argument("--result-json", default="")
    return parser


def latest_result_for_dataset(results_dir: Path, engine: str, dataset: str) -> Path:
    pattern = f"{engine}-{dataset}-search-*-*.json"
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No search result found for pattern: {pattern}")
    return matches[0]


def rebuild_milvus_index(args: argparse.Namespace, experiment: dict, dataset: "Dataset") -> None:
    if experiment.get("engine") != "milvus":
        raise ValueError("--rebuild-index currently supports only Milvus engine.")

    from pymilvus import Collection, connections, wait_for_index_building_complete  # pylint: disable=import-outside-toplevel

    from engine.clients.milvus.config import (  # pylint: disable=import-outside-toplevel
        DISTANCE_MAPPING,
        MILVUS_COLLECTION_NAME,
        MILVUS_DEFAULT_ALIAS,
        MILVUS_DEFAULT_PORT,
    )

    conn_params = dict(experiment.get("connection_params", {}))
    port = str(conn_params.pop("port", MILVUS_DEFAULT_PORT))
    connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        host=args.host,
        port=port,
        **conn_params,
    )

    collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
    try:
        collection.release()
    except Exception:
        pass

    for index in list(collection.indexes):
        index.drop()

    upload_params = dict(experiment.get("upload_params", {}))
    index_type = upload_params.get("index_type", "HNSW")
    index_params = dict(upload_params.get("index_params", {}))
    metric_type = DISTANCE_MAPPING[dataset.config.distance]
    collection.create_index(
        field_name="vector",
        index_params={
            "metric_type": metric_type,
            "index_type": index_type,
            "params": index_params,
        },
    )
    for index in collection.indexes:
        wait_for_index_building_complete(
            MILVUS_COLLECTION_NAME,
            index_name=index.index_name,
            using=MILVUS_DEFAULT_ALIAS,
        )
    collection.load()


def run(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.benchmark_root).resolve()
    sys.path.insert(0, str(benchmark_root))

    from benchmark.config_read import (  # pylint: disable=import-outside-toplevel
        read_dataset_config,
        read_engine_configs,
    )
    from benchmark.dataset import Dataset  # pylint: disable=import-outside-toplevel
    from engine.clients.client_factory import (  # pylint: disable=import-outside-toplevel
        ClientFactory,
    )

    datasets_root = benchmark_root / "datasets"
    dataset_config = None
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = (datasets_root / dataset_path).resolve()

        vectors_path = dataset_path / "vectors.jsonl"
        if vectors_path.exists():
            if args.vector_size <= 0:
                raise ValueError(
                    "--vector-size is required when --dataset-path points to a jsonl dataset directory."
                )
            rel_dataset_path = dataset_path.relative_to(datasets_root)
            dataset_config = {
                "name": args.dataset_name,
                "vector_size": args.vector_size,
                "distance": args.distance,
                "type": "jsonl",
                "path": str(rel_dataset_path),
            }
        else:
            raise FileNotFoundError(f"vectors.jsonl not found in {dataset_path}")
    else:
        all_datasets = read_dataset_config()
        if args.dataset_name not in all_datasets:
            raise KeyError(
                f"Dataset config not found: {args.dataset_name}. "
                "Either provide a valid datasets.json name or pass --dataset-path."
            )
        dataset_config = all_datasets[args.dataset_name]

    dataset = Dataset(dataset_config)
    dataset.download()

    all_engines = read_engine_configs()
    if args.engine_name not in all_engines:
        raise KeyError(f"Engine config not found: {args.engine_name}")
    experiment = all_engines[args.engine_name]

    if args.rebuild_index:
        rebuild_milvus_index(args=args, experiment=experiment, dataset=dataset)

    client = ClientFactory(args.host).build_client(experiment)
    client.run_experiment(
        dataset=dataset,
        skip_upload=args.skip_upload,
        skip_search=args.skip_search,
    )

    result_path = latest_result_for_dataset(
        results_dir=benchmark_root / "results",
        engine=args.engine_name,
        dataset=args.dataset_name,
    )
    payload = {
        "result_file": str(result_path),
        "dataset_name": args.dataset_name,
        "engine_name": args.engine_name,
    }
    print(json.dumps(payload, indent=2))
    if args.result_json:
        result_json_path = Path(args.result_json)
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(build_parser().parse_args())
