#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI: extract PGTuner data_feature row from vector-db-benchmark data (h5 / jsonl / tar; see datasets.json)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PGT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vdb.feature_extract_h5 import extract_row, upsert_data_feature_csv, upsert_milvus_sizes
from vdb.registry import vdb_file_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Name in vector-db-benchmark/datasets/datasets.json")
    p.add_argument("--data-feature-csv", type=Path, default=PGT / "Data" / "vdb_data_feature.csv")
    p.add_argument("--sizes-csv", type=Path, default=PGT / "Data" / "milvus_dataset_sizes.csv")
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override path: .hdf5 file for type=h5, or dataset directory for jsonl/tar",
    )
    p.add_argument("--h5", type=Path, default=None, help="Deprecated alias for --data-path")
    args = p.parse_args()

    override = args.data_path if args.data_path is not None else args.h5
    row = extract_row(args.dataset, data_path=override)
    upsert_data_feature_csv(row, args.data_feature_csv)
    fn = vdb_file_name(args.dataset)
    upsert_milvus_sizes(fn, row["SIZE"], row["q_SIZE"], args.sizes_csv)


if __name__ == "__main__":
    main()
