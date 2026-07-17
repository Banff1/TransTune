import os
import sys

# Ensure auto-configure is in path for configure import (works from any entry point)
_AUTO_CONFIGURE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'auto-configure'))
if _AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.insert(0, _AUTO_CONFIGURE_PATH)
sys.path.append("..")

import joblib
import json
import numpy as np
import time
import subprocess as sp
import random
import traceback
from pathlib import Path
from configure import *

KNOB_PATH = r'/path/to/TransTune/auto-configure/whole_param.json'
RUN_ENGINE_PATH = r'/path/to/TransTune/vector-db-benchmark-master/run_engine_test.sh'
BENCHMARK_CWD = os.path.dirname(RUN_ENGINE_PATH)

def LHS_sample(dimension, num_points, seed):
    np.random.seed(seed)
    
    # Implement Latin Hypercube Sampling manually for older scipy versions
    samples = np.zeros((num_points, dimension))
    
    for d in range(dimension):
        # Generate random permutations for each dimension
        perm = np.random.permutation(num_points)
        # Scale to [0, 1) interval
        samples[:, d] = (perm + np.random.rand(num_points)) / num_points
    
    return samples

class KnobStand:
    def __init__(self, path) -> None:
        self.path = path
        with open(path, 'r') as f:
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name, zero_one_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class StaticEnv:
    def __init__(self, model_path=['XGBoost_20knob_thro.model', 'XGBoost_20knob_prec.model'], knob_path=r'milvus_important_params.json') -> None:
        self.model_path = model_path
        self.get_surrogate(model_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_surrogate(self, surrogate_path):
        # surrogate1, surrogate2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])
        self.model1, self.model2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])

    def get_state(self, knob_vals_arr):
        Y1, Y2 = [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"Index parameters changed: {conf_value}")

            y1 = self.model1.predict([conf_value])[0]
            y2 = self.model2.predict([conf_value])[0]

            self.sampled_times += 1
            print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1} {y2}')
            
            Y1.append(y1)
            Y2.append(y2)
        return np.concatenate((np.array(Y1).reshape(-1,1), np.array(Y2).reshape(-1,1)), axis=1)

class RealEnv:
    def __init__(self, bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH, dataset="glove-100-angular", use_sudo=True) -> None:
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.dataset = dataset
        self.use_sudo = use_sudo  # Set False if user has docker group (no sudo needed)
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y4_record = []
        self.Y_record = []

    def get_state(self, knob_vals_arr):
        # Return metrics: col0=Precisions, col1=RPS, col2=Time (for optimizer compatibility)
        record_log_path = Path(__file__).resolve().parent / "record.log"
        Y1, Y2, Y3, Y4 = [], [], [], []
        for i, record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j, knob_val in enumerate(record)]
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]
            index_conf = dict(zip(index_name, index_value))
            system_conf = dict(zip(system_name, system_value))

            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            try:
                # Use Popen with explicit args (no shell) - ref: auto-configure/transtune/utils.py
                # Python 3.6: use universal_newlines=True (text=True requires 3.7+)
                cmd = ["timeout", "2000", RUN_ENGINE_PATH, "milvus-single-node", "milvus-p10", str(self.dataset)]
                if self.use_sudo:
                    cmd = ["sudo"] + cmd

                process = sp.Popen(
                    cmd,
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    bufsize=1,
                    universal_newlines=True,  # Python 3.6 compatible (text=True in 3.7+)
                    cwd=BENCHMARK_CWD,
                )

                output_lines = []
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    output_lines.append(line)
                process.wait()

                if process.returncode != 0:
                    raise Exception(f"Benchmark failed with return code {process.returncode}")

                result_output = "".join(output_lines)
                lines = result_output.strip().split("\n")
                numeric_values = []

                # Search backwards for result section (handles Chinese output like "测试结果摘要")
                for line in reversed(lines):
                    if "测试结果摘要" in line or "📊" in line or "结果" in line:
                        break
                    for word in line.strip().split():
                        try:
                            numeric_values.append(float(word))
                        except ValueError:
                            continue

                if len(numeric_values) < 3:
                    for item in result_output.strip().split():
                        try:
                            numeric_values.append(float(item))
                        except ValueError:
                            continue

                if len(numeric_values) < 2:
                    raise ValueError(
                        f"Output format unexpected: only found {len(numeric_values)} numeric values. "
                        f"Last 500 chars: {result_output[-500:]}"
                    )

                # Script outputs: mean_precisions rps p95_time (last 3 values)
                if len(numeric_values) >= 3:
                    y1 = numeric_values[-1]   # mean_precisions
                    y2 = numeric_values[-3]   # p95_time
                    y4 = numeric_values[-2]   # rps
                elif len(numeric_values) == 2:
                    y1, y4 = numeric_values[0], numeric_values[1]
                    y2 = 0.1  # p95_time unknown
                else:
                    raise ValueError(f"Unexpected numeric_values length: {len(numeric_values)}")

                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
                self.Y4_record.append(y4)
            except Exception as e:
                print(f"Benchmark failed: {e}")
                traceback.print_exc()
                if self.Y1_record and self.Y4_record:
                    y1, y2, y4 = 0.1, 0.1, 0.1
                else:
                    raise RuntimeError("No previous records and benchmark failed. Cannot continue.") from e

            y3 = int(time.time() - self.t2)
            self.sampled_times += 1
            self.t2 = time.time()
            print(f"[{self.sampled_times}] {int(self.t2 - self.t1)} {y1} {y2} {y3}")

            log_entry = json.dumps({
                "iteration": self.sampled_times,
                "time": int(self.t2 - self.t1),
                "index_conf": index_conf,
                "system_conf": system_conf,
                "precisions": y1,
                "p95time": y2,
                "Time": y3,
                "RPS": y4,
            })
            record_log_path.parent.mkdir(parents=True, exist_ok=True)
            with record_log_path.open("a", encoding="utf-8") as f:
                f.write(log_entry + "\n")

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)
            Y4.append(y4)

        return np.array([Y1, Y4, Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v["default"]) for k, v in self.knob_stand.knobs_detail.items()]

if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

