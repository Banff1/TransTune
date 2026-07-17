import pandas as pd
import json
import sys

def main():
    data_list = []
    
    # 定义列名映射关系 (原始字段路径 -> 目标列名)
    # 嵌套结构使用 . 访问
    mapping = {
        "iteration": "Iteration",
        "time": "Time_Total",
        "index_conf.index_type": "Index_Type",
        "index_conf.nlist": "nlist",
        "index_conf.nprobe": "nprobe",
        "index_conf.m": "m",
        "index_conf.nbits": "nbits",
        "index_conf.M": "M",
        "index_conf.efConstruction": "efConstruction",
        "index_conf.ef": "ef",
        "index_conf.reorder_k": "reorder_k",
        "system_conf.dataCoord*segment*maxSize": "maxSize",
        "system_conf.dataCoord*segment*sealProportion": "sealProportion",
        "system_conf.queryCoord*autoHandoff": "autoHandoff",
        "system_conf.queryCoord*autoBalance": "autoBalance",
        "system_conf.common*gracefulTime": "gracefulTime",
        "system_conf.dataNode*segment*insertBufSize": "insertBufSize",
        "system_conf.rootCoord*minSegmentSizeToEnableIndex": "minSegmentSizeToIndex",
        "precisions": "Precisions",
        "p95time": "p95time",
        "Time": "Time_Step",
        "RPS": "RPS"
    }

    print("请输入数据（按 Ctrl+D 或 Ctrl+Z 结束输入）：", file=sys.stderr)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            # 解析 JSON
            record = json.loads(line)
            # 使用 pd.json_normalize 展平嵌套字段
            flat_df = pd.json_normalize(record, sep='.')
            # 提取转换后的字典
            flat_record = flat_df.to_dict(orient='records')[0]
            
            # 根据 mapping 重新提取和重命名
            processed_data = {}
            for original_key, target_name in mapping.items():
                processed_data[target_name] = flat_record.get(original_key)
            
            data_list.append(processed_data)
        except json.JSONDecodeError:
            continue

    if not data_list:
        print("未检测到有效数据。", file=sys.stderr)
        return

    # 创建 DataFrame 并确保列顺序与要求一致
    df = pd.DataFrame(data_list)
    output_columns = list(mapping.values())
    df = df[output_columns]

    # 保存为 Excel
    output_filename = "arxiv-titles-384-angular-no-filters.xlsx"
    df.to_excel(output_filename, index=False)
    
    print(f"\n处理完成！{len(data_list)} 行数据已保存至: {output_filename}", file=sys.stderr)

if __name__ == "__main__":
    main()