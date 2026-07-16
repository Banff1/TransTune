# -*- coding: utf-8 -*-
"""Load vector-db-benchmark datasets.json and resolve paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
DATASETS_JSON = _REPO / "vector-db-benchmark-master" / "datasets" / "datasets.json"
DATASETS_DIR = _REPO / "vector-db-benchmark-master" / "datasets"


def load_registry() -> List[Dict[str, Any]]:
    with open(DATASETS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset(name: str) -> Dict[str, Any]:
    for d in load_registry():
        if d["name"] == name:
            return d
    raise KeyError(f"Unknown dataset name {name!r}. See {DATASETS_JSON}")


def resolve_local_path(name: str) -> Path:
    """Absolute path to dataset file/dir under vector-db-benchmark-master/datasets/."""
    d = get_dataset(name)
    return (DATASETS_DIR / d["path"]).resolve()


def vdb_file_name(name: str) -> str:
    """Stable FileName column for PGTuner CSVs (must be filesystem-safe)."""
    safe = name.replace("/", "_").replace(" ", "_")
    return f"vdb_{safe}"


def sklearn_metric(distance: str) -> str:
    """Map datasets.json distance to sklearn NearestNeighbors metric."""
    if distance in ("cosine", "angular"):
        return "cosine"
    if distance in ("l2", "euclidean"):
        return "euclidean"
    if distance == "dot":
        return "cosine"
    return "cosine"
