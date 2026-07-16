# vdb-tuning

This repository contains experiments and tools for vector database tuning,
benchmarking, metric extraction, sampling, and configuration optimization.

## Project Structure

- `auto-configure/`: automated tuning and adaptation workflows.
- `pgtuner-configure/`: PostgreSQL or PG-based tuning configuration utilities.
- `ottertune-configure/`: OtterTune-related configuration and tuning code.
- `sampling-configure/`: sampling configuration experiments.
- `vector-db-benchmark-master/`: vector database benchmark resources.
- `extract_metrics.py`: helper script for extracting experiment metrics.
- `test/`: test or validation assets.

## Requirements

Some subprojects include their own dependency files:

- `pgtuner-configure/requirements.txt`
- `ottertune-configure/requirements.txt`
- `vector-db-benchmark-master/pyproject.toml`

Install dependencies from the specific subproject you plan to run.

## Usage

Start by reading the README or scripts inside the relevant subdirectory.
For example, benchmark-related work is likely under
`vector-db-benchmark-master/`, while tuning workflows are under
`auto-configure/`, `pgtuner-configure/`, or `ottertune-configure/`.

## Notes

This file is a lightweight project overview generated without modifying the
existing `README.md` or other project files.
