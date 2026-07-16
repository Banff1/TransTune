#!/usr/bin/env python
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import subprocess as sp

# Add OtterTune server path to Python path
OTTERTUNE_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ottertune/server'))
if OTTERTUNE_SERVER_PATH not in sys.path:
    sys.path.append(OTTERTUNE_SERVER_PATH)

# Add VDTuner auto-configure path to Python path
VDTUNER_AUTO_CONFIGURE_PATH = os.path.abspath('/path/to/TransTune/auto-configure')
if VDTUNER_AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.append(VDTUNER_AUTO_CONFIGURE_PATH)

# Import OtterTune and VDTuner modules
from analysis.gp_tf import GPRGD
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule

# Configuration
KNOB_PATH = '/path/to/TransTune/auto-configure/whole_param.json'
RUN_ENGINE_PATH = '/path/to/TransTune/vector-db-benchmark-master/run_engine_test.sh'
USE_SUDO = True  # Set False if user has docker group (sudo usermod -aG docker $USER)

# Dataset name passed to run_engine_test.sh (same as vector-db-benchmark datasets)
DATASET = "random-100-match-kw-small-vocab-no-filters"

# Main tuning log (under ottertune-configure)
_LOG_DIR = "/path/to/TransTune/ottertune-configure/log"
LOG_FILE = os.path.join(_LOG_DIR, f"{DATASET}_ottertune.log")

# Per-iteration JSON lines (same style as vdtuner_interface RealEnv.get_state)
RECORD_LOG_PATH = Path(_LOG_DIR) / "record.log"
# Set True to empty record.log once at program start (default: append for a continuous log)
CLEAR_RECORD_LOG_ON_START = False


def run_engine_test(dataset=None, use_sudo=None):
    """Run the engine test and return (rps, precision, p95_time) or (None, None, None).

    Parsing matches run_engine_test.sh summary: three numbers with indices depending on scan order;
    see comments below (aligned with vdtuner_interface RealEnv.get_state).
    """
    if dataset is None:
        dataset = DATASET
    if use_sudo is None:
        use_sudo = USE_SUDO
    try:
        cmd = ["timeout", "900", RUN_ENGINE_PATH, "milvus-single-node", "milvus-p10", str(dataset)]
        if use_sudo:
            cmd = ["sudo"] + cmd
        result = sp.run(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=os.path.dirname(RUN_ENGINE_PATH),
        )
        result_output = result.stdout.decode(errors='ignore') + result.stderr.decode(errors='ignore')
        lines = result_output.strip().split('\n')

        # Search backwards for result section (ref: auto-configure RealEnv.get_state)
        # Script prints one number per line after "📊 测试结果摘要:" (mean_precisions, rps, p95 order from grep)
        numeric_values = []
        for line in reversed(lines):
            if '测试结果摘要' in line or '📊' in line or '结果' in line:
                break
            for word in line.strip().split():
                try:
                    numeric_values.append(float(word))
                except ValueError:
                    continue

        # Fallback: extract all numeric values from entire output
        from_reversed = len(numeric_values) >= 3
        if not from_reversed:
            numeric_values = []
            for item in result_output.strip().split():
                try:
                    numeric_values.append(float(item))
                except ValueError:
                    continue

        if len(numeric_values) < 2:
            return None, None, None

        # Same indexing as historical main_ottertune / reversed-block convention:
        # from_reversed: [-3]=p95, [-2]=rps, [-1]=precision
        # fallback full scan: [-3]=precision, [-2]=rps, [-1]=p95
        if len(numeric_values) >= 3:
            if from_reversed:
                p95_time = numeric_values[-3]
                rps = numeric_values[-2]
                precision = numeric_values[-1]
            else:
                precision = numeric_values[-3]
                rps = numeric_values[-2]
                p95_time = numeric_values[-1]
        else:
            rps = numeric_values[-2]
            precision = numeric_values[-1]
            p95_time = None

        if 0 <= precision <= 1 and rps > 0:
            return rps, precision, p95_time
        return None, None, None
    except Exception as e:
        print(f"Error running engine test: {e}")
        return None, None, None

def log(message):
    """Log message to file and print"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')


def append_record_line(
    *,
    iteration: int,
    time_cumulative_sec: int,
    index_conf: dict,
    system_conf: dict,
    precisions,
    p95time,
    bench_duration_sec: int,
    rps,
):
    """Append one JSON line to record.log (same schema as vdtuner/record.log lines)."""
    RECORD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Key order matches typical tooling / human reading
    row = {
        "iteration": int(iteration),
        "time": int(time_cumulative_sec),
        "index_conf": index_conf,
        "system_conf": system_conf,
        "precisions": None if precisions is None else float(precisions),
        "p95time": None if p95time is None else float(p95time),
        "Time": int(bench_duration_sec),
        "RPS": None if rps is None else float(rps),
    }
    with RECORD_LOG_PATH.open("a", encoding="utf-8") as f:
        # default=str: tolerate numpy scalars inside index_conf / system_conf if any
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main():
    t_main_start = time.time()
    if CLEAR_RECORD_LOG_ON_START:
        RECORD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_LOG_PATH.write_text("", encoding="utf-8")
        log(f"Truncated {RECORD_LOG_PATH} (CLEAR_RECORD_LOG_ON_START=True)")
    log("--- OtterTune VDTuner Integration ---\n")
    log(f"Dataset: {DATASET} | record.log: {RECORD_LOG_PATH}")

    record_iter = 0  # 1-based global line counter for record.log (initial + GPRGD rounds)

    # 1. Initialize environment and knob stand
    log("Initializing environment...")
    env = RealEnv(bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH, dataset=DATASET, use_sudo=USE_SUDO)
    knob_stand = KnobStand(KNOB_PATH)
    num_knobs = len(knob_stand.knobs_detail.keys())
    log(f"Number of knobs: {num_knobs}")

    # 2. Initial sampling using LHS
    log("\n--- Initial Sampling ---\n")
    num_initial_samples = 7
    log(f"Generating {num_initial_samples} initial samples using Latin Hypercube Sampling...")
    
    X_train = LHS_sample(num_knobs, num_initial_samples, seed=1)
    log(f"Initial samples shape: {X_train.shape}")

    # 3. Run initial samples to get performance data
    X_success, y_success = [], []
    for i, sample in enumerate(X_train):
        log(f"\nRunning initial sample {i+1}/{num_initial_samples}...")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], sample[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Index configuration: {index_conf}")
        log(f"System configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        t_bm = time.time()
        rps, precision, p95_time = run_engine_test()
        bench_dur = int(time.time() - t_bm)
        objective = None
        if rps is not None and precision is not None:
            # Single objective: maximize RPS when precision >= 0.9, else penalize
            if precision >= 0.9:
                objective = -rps  # We want to minimize, so negative RPS
            else:
                objective = 1e6  # Large penalty for low precision

            X_success.append(sample)
            y_success.append(objective)
            p95_str = f"{p95_time:.6f}" if p95_time is not None else "n/a"
            log(
                f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, "
                f"p95_time = {p95_str}, Objective = {objective:.4f}"
            )
        else:
            log("Engine test failed, skipping this sample")

        record_iter += 1
        append_record_line(
            iteration=record_iter,
            time_cumulative_sec=int(time.time() - t_main_start),
            index_conf=index_conf,
            system_conf=system_conf,
            precisions=precision,
            p95time=p95_time,
            bench_duration_sec=bench_dur,
            rps=rps,
        )
    
    X_train = np.array(X_success, dtype=np.float32) if X_success else np.zeros((0, num_knobs), dtype=np.float32)
    y_train = np.array(y_success, dtype=np.float32).reshape(-1, 1) if y_success else np.zeros((0, 1), dtype=np.float32)
    log(f"\nInitial training data shape: X={X_train.shape}, y={y_train.shape}")

    if len(X_success) == 0:
        log("ERROR: All initial samples failed. Check Docker permissions (add user to docker group) and ensure Milvus can start.")
        log("  Fix: sudo usermod -aG docker $USER  (then log out and back in)")
        return

    # 4. Scale performance data
    y_min = np.min(y_train)
    y_max = np.max(y_train)
    y_train_scaled = (y_train - y_min) / (y_max - y_min) if y_max > y_min else y_train

    # 5. Initialize OtterTune GPRGD model
    log("\n--- Initializing OtterTune GPRGD Model ---\n")
    model = GPRGD(
        length_scale=1.0,
        magnitude=1.0,
        ridge=0.1,
        max_iter=20,
        learning_rate=0.001,
        check_numerics=True,
        debug=True
    )

    # 6. Run iterative tuning
    log("\n--- Iterative Tuning ---\n")
    num_iterations = 250 - num_initial_samples
    
    X_min = np.zeros(num_knobs, dtype=np.float32)
    X_max = np.ones(num_knobs, dtype=np.float32)
    
    for iteration in range(num_iterations):
        log(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # Fit model to current data
        log("Fitting GPRGD model...")
        model.fit(X_train, y_train_scaled, X_min=X_min, X_max=X_max)
        
        # Generate new configuration candidates
        log("Generating new configuration candidates...")
        # Use LHS to generate candidate points
        num_candidates = 10
        X_candidates = LHS_sample(num_knobs, num_candidates, seed=iteration+2)
        
        # Predict objective for candidates
        log("Predicting objective for candidates...")
        results = model.predict(X_candidates)
        
        # Select best candidate: use minl (predicted loss), not minl_conf (config values)
        # minl_conf is (n_candidates, n_feats) - argmin on it returns flattened index (wrong!)
        minl_flat = np.asarray(results.minl).ravel()
        best_idx = int(np.argmin(minl_flat))
        best_candidate = results.minl_conf[best_idx]  # Use optimized config from gradient descent
        log(f"Best candidate found: {best_candidate}")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], best_candidate[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Best index configuration: {index_conf}")
        log(f"Best system configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        log("Running engine test with best candidate...")
        t_bm = time.time()
        rps, precision, p95_time = run_engine_test()
        bench_dur = int(time.time() - t_bm)
        objective = None

        if rps is not None and precision is not None:
            # Calculate objective
            if precision >= 0.8:
                objective = -rps
            else:
                objective = 1e6

            # Scale objective
            objective_scaled = (objective - y_min) / (y_max - y_min) if y_max > y_min else objective

            # Add to training data
            X_train = np.vstack((X_train, best_candidate.reshape(1, -1)))
            y_train = np.vstack((y_train, np.array([[objective]])))
            y_train_scaled = np.vstack((y_train_scaled, np.array([[objective_scaled]])))

            p95_str = f"{p95_time:.6f}" if p95_time is not None else "n/a"
            log(
                f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, "
                f"p95_time = {p95_str}, Objective = {objective:.4f}"
            )
            log(f"Updated training data shape: X={X_train.shape}, y={y_train.shape}")
        else:
            log("Engine test failed, skipping this candidate")

        record_iter += 1
        append_record_line(
            iteration=record_iter,
            time_cumulative_sec=int(time.time() - t_main_start),
            index_conf=index_conf,
            system_conf=system_conf,
            precisions=precision,
            p95time=p95_time,
            bench_duration_sec=bench_dur,
            rps=rps,
        )
        
        # Update min and max for scaling
        y_min = np.min(y_train)
        y_max = np.max(y_train)

    log("\n--- Tuning Complete ---")
    log(f"Final training data shape: X={X_train.shape}, y={y_train.shape}")

if __name__ == '__main__':
    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write('')
    
    main()
