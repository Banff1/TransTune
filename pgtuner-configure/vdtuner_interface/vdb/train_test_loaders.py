# -*- coding: utf-8 -*-
"""
Load train (base) / test (query) vectors the same way vector-db-benchmark does.

- h5: ``train`` / ``test`` datasets in a single .hdf5 file (AnnH5Reader).
- jsonl: directory with ``vectors.jsonl`` + ``queries.jsonl`` (JSONReader).
- tar: directory with ``vectors.npy`` + ``tests.jsonl`` (AnnCompoundReader; each line has ``query``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np

from .registry import get_dataset, resolve_local_path

TrainArray = Union[np.ndarray, np.memmap]


def _load_h5_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "train" not in f or "test" not in f:
            raise ValueError(f"Expected HDF5 datasets 'train' and 'test' in {path}")
        train = np.asarray(f["train"][:], dtype=np.float32)
        test = np.asarray(f["test"][:], dtype=np.float32)
    return train, test


def _load_vectors_jsonl(path: Path) -> np.ndarray:
    vectors: list = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            vectors.append(json.loads(line))
    if not vectors:
        raise ValueError(f"No vectors in {path}")
    return np.asarray(vectors, dtype=np.float32)


def _load_jsonl_dir(d: Path) -> Tuple[np.ndarray, np.ndarray]:
    vf = d / "vectors.jsonl"
    qf = d / "queries.jsonl"
    if not vf.is_file():
        raise FileNotFoundError(
            f"Missing {vf}. JSONL datasets need vectors.jsonl and queries.jsonl under the dataset path."
        )
    if not qf.is_file():
        raise FileNotFoundError(
            f"Missing {qf}. JSONL datasets need vectors.jsonl and queries.jsonl under the dataset path."
        )
    return _load_vectors_jsonl(vf), _load_vectors_jsonl(qf)


def _load_tar_queries_from_tests(tests_path: Path) -> np.ndarray:
    rows: list = []
    with open(tests_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if "query" not in j:
                raise ValueError(f"Expected key 'query' in each line of {tests_path} (AnnCompoundReader format).")
            rows.append(j["query"])
    if not rows:
        raise ValueError(f"No queries in {tests_path}")
    return np.asarray(rows, dtype=np.float32)


def _load_tar_dir(d: Path) -> Tuple[TrainArray, np.ndarray]:
    vf = d / "vectors.npy"
    tf = d / "tests.jsonl"
    if not vf.is_file():
        raise FileNotFoundError(
            f"Missing {vf}. Tar-type datasets use AnnCompoundReader layout: vectors.npy + tests.jsonl."
        )
    if not tf.is_file():
        raise FileNotFoundError(
            f"Missing {tf}. Tar-type datasets use AnnCompoundReader layout: vectors.npy + tests.jsonl."
        )
    train = np.load(vf, mmap_mode="r")
    test = _load_tar_queries_from_tests(tf)
    return train, test


def load_train_test(
    dataset_name: str,
    *,
    data_path: Optional[Path] = None,
) -> Tuple[TrainArray, np.ndarray]:
    """
    Return (train, test). Train may be ndarray or memmap (large ``vectors.npy``).
    """
    meta = get_dataset(dataset_name)
    root = Path(data_path).resolve() if data_path else resolve_local_path(dataset_name)
    typ = meta.get("type")

    if typ == "h5":
        p = root
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing HDF5 file: {p}\nDownload via vector-db-benchmark Dataset.download() or place the file."
            )
        return _load_h5_file(p)

    if typ == "jsonl":
        d = root if root.is_dir() else root.parent
        if not d.is_dir():
            raise FileNotFoundError(f"JSONL dataset path must be a directory: {d}")
        return _load_jsonl_dir(d)

    if typ == "tar":
        d = root if root.is_dir() else root.parent
        if not d.is_dir():
            raise FileNotFoundError(f"Tar dataset path must be a directory: {d}")
        return _load_tar_dir(d)

    raise ValueError(f"Unsupported dataset type {typ!r} for {dataset_name!r} in datasets.json")
