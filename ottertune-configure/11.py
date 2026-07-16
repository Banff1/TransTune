import re
import json
from datetime import datetime

def process_log_line_by_line(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按照迭代轮次进行分割
    blocks = re.split(r'--- Iteration \d+/\d+ ---', content)
    
    results = []
    first_start_time = None

    # blocks[0] 通常是分割符前的内容，从索引 1 开始处理每一轮
    for i in range(1, len(blocks)):
        block = blocks[i].strip()
        if not block or "Engine test failed" in block:
            continue

        try:
            # 1. 提取所有时间戳用于计算
            timestamps = re.findall(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', block)
            if not timestamps:
                continue
            
            start_dt = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S')
            
            if first_start_time is None:
                first_start_time = start_dt
            
            # 计算 Time (本轮秒数) 和 time (总累计秒数)
            current_iter_duration = int((end_dt - start_dt).total_seconds())
            total_elapsed_time = int((end_dt - first_start_time).total_seconds())

            # 2. 提取配置字符串
            index_conf_match = re.search(r'Best index configuration: ({.*?})', block)
            system_conf_match = re.search(r'Best system configuration: ({.*?})', block)
            
            if not index_conf_match or not system_conf_match:
                continue

            # 处理 Python 字典转 JSON 对象 (处理单引号和布尔值)
            def to_json_dict(s):
                s = s.replace("'", '"').replace("True", "true").replace("False", "false")
                return json.loads(s)

            # 3. 提取性能指标
            rps = float(re.search(r'RPS = ([\d.]+)', block).group(1))
            precision = float(re.search(r'Precision = ([\d.]+)', block).group(1))

            # 构建本轮数据
            data = {
                "iteration": i,
                "time": total_elapsed_time,
                "index_conf": to_json_dict(index_conf_match.group(1)),
                "system_conf": to_json_dict(system_conf_match.group(1)),
                "precisions": precision,
                "Time": current_iter_duration,
                "RPS": rps
            }
            results.append(data)
        except Exception:
            continue

    # --- 关键输出部分：每轮数据占一行 ---
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for index, item in enumerate(results):
            line = json.dumps(item, ensure_ascii=False)
            # 如果不是最后一轮，末尾加逗号
            comma = "," if index < len(results) - 1 else ""
            f.write(f"  {line}{comma}\n")
        f.write("]\n")
    
    print(f"处理完成！有效数据：{len(results)} 轮。文件已保存至: {output_path}")

# 执行
process_log_line_by_line('Iteration.txt', 'Iteration_line_by_line.json')