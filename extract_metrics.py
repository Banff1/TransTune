#!/usr/bin/env python3
import json
import os
from datetime import datetime


RESULTS_DIR = "/path/to/TransTune/vector-db-benchmark-master/results"
PREFIX = "milvus-p10-arxiv-titles-384-angular-no-filters-search-0-"
SUFFIX = ".json"

START_NAME = (
    "milvus-p10-arxiv-titles-384-angular-no-filters-search-0-2026-01-15-10-13-13"
)
END_NAME = (
    "milvus-p10-arxiv-titles-384-angular-no-filters-search-0-2026-01-19-13-30-20"
)


def parse_dt_from_name(filename):
    if not filename.startswith(PREFIX) or not filename.endswith(SUFFIX):
        return None
    ts = filename[len(PREFIX) : -len(SUFFIX)]
    try:
        return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def main():
    start_dt = parse_dt_from_name(START_NAME + SUFFIX)
    end_dt = parse_dt_from_name(END_NAME + SUFFIX)
    if start_dt is None or end_dt is None:
        raise ValueError("Invalid START_NAME or END_NAME.")

    files = []
    for name in os.listdir(RESULTS_DIR):
        dt = parse_dt_from_name(name)
        if dt is None:
            continue
        if start_dt <= dt <= end_dt:
            files.append((dt, name))

    files.sort(key=lambda item: item[0])

    for _, name in files:
        path = os.path.join(RESULTS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        mean_precisions = results.get("mean_precisions", "")
        p95_time = results.get("p95_time", "")
        rps = results.get("rps", "")
        print(f"{mean_precisions}\t{p95_time}\t{rps}")


if __name__ == "__main__":
    main()

