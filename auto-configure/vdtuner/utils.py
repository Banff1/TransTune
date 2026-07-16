import sys 
sys.path.append("..") 

import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as sp
import random
from configure import *
import threading
import traceback
from pathlib import Path
from typing import Optional

VDB_TUNING_ROOT = "/path/to/TransTune"
KNOB_PATH = f"{VDB_TUNING_ROOT}/auto-configure/whole_param.json"
RUN_ENGINE_PATH = f"{VDB_TUNING_ROOT}/vector-db-benchmark-master/run_engine_test.sh"

def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)

    return latin_samples

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
    def __init__(
        self,
        bench_path=RUN_ENGINE_PATH,
        knob_path=KNOB_PATH,
        dataset="glove-100-angular",
        record_log_path: Optional[str] = None,
    ) -> None:
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.dataset = dataset  # Store the dataset name
        default_record_log_path = Path(__file__).resolve().parent / "record.log"
        self.record_log_path = Path(record_log_path) if record_log_path else default_record_log_path
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y4_record = []
        self.Y_record = []

    def get_state(self, knob_vals_arr):
        # Return metrics in the following order (for the optimizer):
        # - col 0: Precisions (larger is better)
        # - col 1: RPS        (larger is better)
        # - col 2: Time (optional bookkeeping; not used by current optimizer)
        Y1, Y2, Y3, Y4 = [], [], [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(record)]
            
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]

            index_conf = dict(zip(index_name,index_value))
            system_conf = dict(zip(system_name,system_value))

            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            # print(f"Parameters changed to: {index_conf} {system_conf}")

            try:
                print(f"--- DEBUG: Starting command:---", flush=True)
                # Avoid `shell=True` to prevent shell parsing errors and make args unambiguous.
                process = sp.Popen(
                    ["sudo", "timeout", "2000", RUN_ENGINE_PATH, "milvus-single-node", "milvus-p10", str(self.dataset)],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,  # 将错误和标准输出合并，方便实时打印
                    text=True,
                    bufsize=1,         # 行缓冲
                    universal_newlines=True,
                )

                output_lines = []

                # 实时读取子进程的输出
                for line in process.stdout:
                    # 1. 打印到当前终端（这样 nohup 日志就能实时收到了）
                    sys.stdout.write(line)
                    sys.stdout.flush() 
                    # 2. 存入变量供后续解析
                    output_lines.append(line)

                # 等待子进程结束
                process.wait()

                if process.returncode != 0:
                    raise Exception(f"Benchmark failed with return code {process.returncode}")

                result_output = "".join(output_lines)
                lines = result_output.strip().split('\n')   
                
                # The script outputs results at the end: "📊 测试结果摘要:" followed by three numbers
                # Read from the end backwards, extract numeric values until we hit the result summary line
                numeric_values = []
                
                # Search from the end backwards for the result section
                for line in reversed(lines):
                    # Stop when we encounter the result summary line (don't process this line)
                    if '测试结果摘要' in line or '📊' in line or '结果' in line:
                        break
                    # Extract all numeric values from this line
                    print("line1: ", line)
                    words = line.strip().split()
                    for word in words:
                        try:
                            numeric_values.append(float(word))
                        except ValueError:
                            continue
                
                # Fallback: if we didn't find the result section, extract all numeric values from the end
                if len(numeric_values) < 3:
                    result_list = result_output.strip().split()
                    numeric_values = []
                    for item in result_list:
                        try:
                            numeric_values.append(float(item))
                        except ValueError:
                            continue
                
                if len(numeric_values) < 2:
                    print(f"Warning: Could not extract enough numeric values from output.")
                    print(f"Output (last 500 chars): {result_output[-500:]}")
                    print(f"Extracted numeric values: {numeric_values}")
                    raise ValueError(f"Output format unexpected: only found {len(numeric_values)} numeric values")

                print("numeric_values: ", numeric_values)
                # The script outputs: mean_precisions rps p95_time
                # We want: y1 = p95_time (latency), y2 = mean_precisions (recall/precision)
                if len(numeric_values) >= 3:
                    # Take the last 3 values: [mean_precisions, rps, p95_time]
                    # Index: -3 is p95_time, -2 is rps, -1 is mean_precisions
                    y1, y2, y4 = numeric_values[-1], numeric_values[-3], numeric_values[-2] # mean_precisions, p95_time, rps
                elif len(numeric_values) == 2:
                    # Fallback: assume last two are rps and p95_time
                    y1, y2 = numeric_values[-1], numeric_values[0]  # p95_time, rps (fallback)
                else:
                    raise ValueError(f"Unexpected number of numeric values: {len(numeric_values)}")
                
                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
                self.Y4_record.append(y4)
            except Exception as e:
                print(f"sp.run failed: {e}")
                traceback.print_exc()
                if len(self.Y1_record) > 0 and len(self.Y2_record) > 0:
                    # y1, y2 = min(self.Y1_record), min(self.Y2_record)
                    y1, y2, y4 = 0.1, 0.1, 0.1
                else:
                    print("Error: No previous records available and benchmark failed. Using default values.")
                    y1, y2, y4 = 0.1, 0.1, 0.1  # Default fallback values
            
            y3 = int(time.time()-self.t2)
            self.sampled_times += 1

            self.t2 = time.time()
            print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
            # 使用 JSON 格式避免 shell 语法错误，并转义特殊字符
            import json
            log_entry = json.dumps({
                'iteration': self.sampled_times,
                'time': int(self.t2-self.t1),
                'index_conf': index_conf,
                'system_conf': system_conf,
                'precisions': y1,
                'p95time': y2,
                'Time': y3,
                'RPS': y4
            })
            # Write via Python to avoid `/bin/sh` syntax errors when log entry contains special chars.
            self.record_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_log_path.open("a", encoding="utf-8") as f:
                f.write(log_entry + "\n")

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)
            Y4.append(y4)

        # NOTE: y2 is p95time (smaller is better), but we return RPS for optimization.
        # Keep p95time only in logs; optimizer uses (Precisions, RPS).
        return np.array([Y1, Y4, Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

