# TransTune

This repository contains the code and configuration used for vector database tuning experiments, including benchmark execution, offline tuning, online tuning with priors, and data-drift adaptation.

## Project Structure

- `auto-configure/transtune/`: TransTune tuning workflows.
- `auto-configure/transtune/prior/`: prior configuration files for online tuning.
- `auto-configure/transtune/new_adapt/`: data-drift simulation and adaptive tuning workflows.
- `vector-db-benchmark-master/`: vector database benchmark runner and Milvus benchmark configuration.
- `pgtuner-configure/`, `ottertune-configure/`, `sampling-configure/`: auxiliary tuning and sampling components.

## Environment Setup

### 1. Install Milvus

Install a standalone Milvus instance with Docker Compose by following the official Milvus guide:

https://milvus.io/docs/install_standalone-docker-compose.md

The benchmark scripts in this repository assume a local Milvus deployment and use the Milvus single-node configuration under:

```bash
vector-db-benchmark-master/engine/servers/milvus-single-node/
```

### 2. Configure Python

Create and activate a Python virtual environment. Python 3.11 or newer is recommended.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

The experimental environment used for this repository includes the following key packages:

```bash
pip install botorch gpytorch torch pyro-ppl numpy scipy pandas scikit-learn h5py pymilvus qdrant-client redis opensearch-py elasticsearch typer rich tqdm pyyaml requests matplotlib statsmodels openpyxl
```

Key package groups are:

- Bayesian optimization and surrogate modeling: `botorch`, `gpytorch`, `torch`, `pyro-ppl`.
- Data processing and model training: `numpy`, `scipy`, `pandas`, `scikit-learn`, `h5py`, `statsmodels`.
- Vector database clients and benchmark backends: `pymilvus`, `qdrant-client`, `redis`, `opensearch-py`, `elasticsearch`.
- CLI, logging, and utility dependencies: `typer`, `rich`, `tqdm`, `pyyaml`, `requests`.

Some subprojects also provide their own dependency files, such as:

```bash
pgtuner-configure/requirements.txt
ottertune-configure/requirements.txt
vector-db-benchmark-master/pyproject.toml
```

## Running the Repository

### 1. Run a Milvus benchmark

From the repository root, enter the benchmark directory and run a single benchmark workload:

```bash
cd vector-db-benchmark-master
./run_engine_test.sh milvus-single-node milvus-p10 random-100-match-kw-small-vocab-no-filters
cd ..
```

The third argument specifies the dataset/workload. Replace `random-100-match-kw-small-vocab-no-filters` with another configured dataset name when needed.

### 2. Run offline tuning

```bash
cd auto-configure/transtune
python main_tuner.py
```

### 3. Run online tuning with priors

```bash
cd auto-configure/transtune
python main_tuner_priors.py --prior-config prior/random-range-2048-angular-no-filters.json
```

### 4. Run data-drift adaptive tuning

```bash
cd auto-configure/transtune/new_adapt
python run_drift_adaptive_pipeline_bo.py
```

This workflow simulates a data-drift setting and runs adaptive tuning under the drifted workload.
