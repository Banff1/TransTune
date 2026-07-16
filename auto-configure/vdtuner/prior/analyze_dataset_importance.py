#!/usr/bin/env python3
"""
分析数据集，确定：
1. 较好的index type：统计每个type对应的平均p95time，保留3个最好的（不统计缺失值）
2. 重要旋钮：训练随机森林模型预测p95time，选择重要性排在前10个旋钮
3. 旋钮范围：针对RPS在超出平均RPS 30%以上的配置下，统计重要旋钮的取值范围

输入：数据集文件xlsx
输出：较好的index type，重要旋钮，旋钮范围
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json


# ==================== 配置区域 ====================
# 指定要分析的数据集文件
DATASET_FILE = "random-range-2048-angular-no-filters-low-sim.xlsx"
# 输出JSON文件路径（如果为None则不保存）
OUTPUT_JSON = None
# 例如: "analysis_results.json"

# Precisions阈值（用于分析旋钮范围时的过滤条件）
PRECISION_THRESHOLD = 0.90  # 只统计Precisions大于此值的配置

#高于平均RPS比例, 在此之上的才作为旋钮范围的依据
RPS_THRESHOLD = 1.1

# 数据目录
DATA_DIR = "/path/to/TransTune/auto-configure/vdtuner/prior"
# ==================================================


def load_dataset(file_path):
    """加载数据集文件"""
    print(f"正在加载数据集: {file_path}")
    try:
        df = pd.read_excel(file_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df
    except ImportError as e:
        raise ImportError(
            "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
        ) from e
    except Exception as e:
        raise Exception(f"读取文件时出错: {e}") from e


def analyze_index_types(df):
    """
    分析较好的index type
    统计每个type对应的平均p95time，保留3个最好的（不统计缺失值）
    """
    print("\n" + "="*60)
    print("1. 分析较好的Index Type")
    print("="*60)
    
    if 'Index_Type' not in df.columns:
        print("警告: 未找到 'Index_Type' 列")
        return []
    
    if 'p95time' not in df.columns:
        print("警告: 未找到 'p95time' 列")
        return []
    
    # 过滤掉缺失值（如果一行中任何列有缺失值，则删除该行）
    df_clean = df[['Index_Type', 'p95time']].dropna(how='any')
    
    if len(df_clean) == 0:
        print("警告: 没有有效的数据（所有数据都包含缺失值）")
        return []
    
    # 按Index_Type分组，计算平均p95time
    index_type_stats = df_clean.groupby('Index_Type')['p95time'].agg(['mean', 'count']).reset_index()
    index_type_stats.columns = ['Index_Type', 'avg_p95time', 'count']
    
    # 按平均p95time排序（越小越好）
    index_type_stats = index_type_stats.sort_values('avg_p95time', ascending=True)
    
    # 保留3个最好的
    top_3_index_types = index_type_stats.head(3)
    
    print(f"\n所有Index Type统计:")
    print(index_type_stats.to_string(index=False))
    print(f"\n前3个最好的Index Type (按平均p95time排序，越小越好):")
    print(top_3_index_types.to_string(index=False))
    
    return top_3_index_types['Index_Type'].tolist()


def analyze_important_knobs(df):
    """
    分析重要旋钮
    训练随机森林模型预测p95time，选择重要性排在前10个旋钮
    """
    print("\n" + "="*60)
    print("2. 分析重要旋钮")
    print("="*60)
    
    if 'p95time' not in df.columns:
        raise ValueError("未找到 'p95time' 列")
    
    # 定义特征列（旋钮）
    knob_columns = [
        'Index_Type', 'nlist', 'nprobe', 'm', 'nbits', 'M', 
        'efConstruction', 'ef', 'reorder_k', 'maxSize', 
        'sealProportion', 'autoHandoff', 'autoBalance', 
        'gracefulTime', 'insertBufSize', 'minSegmentSizeToIndex'
    ]
    
    # 检查哪些列存在
    available_knobs = [col for col in knob_columns if col in df.columns]
    
    if len(available_knobs) == 0:
        raise ValueError(f"未找到任何旋钮列。可用列: {df.columns.tolist()}")
    
    print(f"找到 {len(available_knobs)} 个旋钮列: {available_knobs}")
    
    # 准备数据：删除包含缺失值的行（如果一行中任何列有缺失值，则删除该行）
    feature_cols = available_knobs + ['p95time']
    before_drop = len(df)
    df_clean = df[feature_cols + ['RPS']].dropna(how='any')
    after_drop = len(df_clean)
    
    if after_drop == 0:
        raise ValueError("删除缺失值后没有有效数据")
    
    print(f"原始数据行数: {before_drop}")
    print(f"删除包含缺失值的行后，有效数据行数: {after_drop}")
    if before_drop > after_drop:
        print(f"已删除 {before_drop - after_drop} 行包含缺失值的数据")
    
    # 分离特征和目标
    X = df_clean[available_knobs].copy()
    y = df_clean['p95time'].values
    
    # 处理分类特征
    label_encoders = {}
    for col in available_knobs:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # 训练随机森林模型
    print("\n正在训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'knob': available_knobs,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n所有旋钮的重要性排序:")
    print(feature_importance.to_string(index=False))
    
    # 选择前10个重要旋钮
    top_10_knobs = feature_importance.head(10)['knob'].tolist()
    
    print(f"\n前10个重要旋钮:")
    print(feature_importance.head(10).to_string(index=False))
    
    return top_10_knobs, feature_importance


def analyze_knob_ranges(df, important_knobs, precision_threshold=0.80):
    """
    分析旋钮范围
    1. 首先过滤出Precisions大于阈值的配置
    2. 在这些配置中计算平均RPS
    3. 找出RPS超出平均RPS 30%以上的配置
    4. 统计这些配置中重要旋钮的取值范围
    
    Args:
        df: 数据框
        important_knobs: 重要旋钮列表
        precision_threshold: Precisions阈值，默认0.80
    """
    print("\n" + "="*60)
    print("3. 分析重要旋钮的取值范围")
    print("="*60)
    
    if 'RPS' not in df.columns:
        raise ValueError("未找到 'RPS' 列")
    
    if 'Precisions' not in df.columns:
        raise ValueError("未找到 'Precisions' 列")
    
    # 过滤掉缺失值（如果一行中任何列有缺失值，则删除该行）
    before_drop = len(df)
    required_cols = important_knobs + ['RPS', 'Precisions']
    df_clean = df[required_cols].dropna(how='any')
    after_drop = len(df_clean)
    
    if after_drop == 0:
        raise ValueError("删除缺失值后没有有效数据")
    
    if before_drop > after_drop:
        print(f"删除包含缺失值的行: {before_drop - after_drop} 行，剩余有效数据: {after_drop} 行")
    
    # 第一步：筛选Precisions大于阈值的配置
    df_high_precision = df_clean[df_clean['Precisions'] > precision_threshold].copy()
    print(f"\nPrecisions阈值: {precision_threshold}")
    print(f"Precisions > {precision_threshold} 的配置数量: {len(df_high_precision)}")
    
    if len(df_high_precision) == 0:
        print(f"警告: 没有配置满足 Precisions > {precision_threshold} 的条件")
        return {}
    
    # 第二步：在这些配置中计算平均RPS
    avg_rps = df_high_precision['RPS'].mean()
    print(f"Precisions > {precision_threshold} 的配置中，平均RPS: {avg_rps:.2f}")
    
    # 第三步：筛选RPS超出平均RPS 30%以上的配置
    threshold_rps = avg_rps * RPS_THRESHOLD
    df_filtered = df_high_precision[df_high_precision['RPS'] >= threshold_rps].copy()
    
    print(f"RPS阈值 (平均RPS * {RPS_THRESHOLD}): {threshold_rps:.2f}")
    print(f"满足条件的配置数量 (Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f}): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print(f"警告: 没有配置满足 Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f} 的条件")
        return {}
    
    # 第四步：统计每个重要旋钮的取值范围
    knob_ranges = {}
    
    for knob in important_knobs:
        if knob not in df_filtered.columns:
            continue
        
        knob_values = df_filtered[knob]
        
        # 如果是数值型
        if pd.api.types.is_numeric_dtype(knob_values):
            knob_ranges[knob] = {
                'type': 'numeric',
                'min': float(knob_values.min()),
                'max': float(knob_values.max()),
                'mean': float(knob_values.mean()),
                'median': float(knob_values.median()),
                'unique_values': sorted(knob_values.unique().tolist())
            }
        else:
            # 如果是分类型
            unique_values = knob_values.unique().tolist()
            knob_ranges[knob] = {
                'type': 'categorical',
                'unique_values': sorted([str(v) for v in unique_values]),
                'value_counts': knob_values.value_counts().to_dict()
            }
    
    # 打印结果
    print(f"\n重要旋钮的取值范围 (Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f}):")
    for knob, range_info in knob_ranges.items():
        print(f"\n{knob}:")
        if range_info['type'] == 'numeric':
            print(f"  类型: 数值型")
            print(f"  最小值: {range_info['min']}")
            print(f"  最大值: {range_info['max']}")
            print(f"  平均值: {range_info['mean']:.2f}")
            print(f"  中位数: {range_info['median']:.2f}")
            print(f"  唯一值数量: {len(range_info['unique_values'])}")
            if len(range_info['unique_values']) <= 20:
                print(f"  唯一值: {range_info['unique_values']}")
        else:
            print(f"  类型: 分类型")
            print(f"  唯一值: {range_info['unique_values']}")
            print(f"  值分布: {range_info['value_counts']}")
    
    return knob_ranges


def find_top_rps_configs(df, precision_threshold=0.80, top_n=10):
    """
    找出precision大于阈值且RPS排在前N的配置
    
    Args:
        df: 数据框
        precision_threshold: Precisions阈值
        top_n: 返回前N个配置，默认10
    
    Returns:
        包含Index_Type, Precisions, p95time, RPS的前N个配置
    """
    print("\n" + "="*60)
    print("4. 找出Precision大于阈值且RPS排在前10的配置")
    print("="*60)
    
    required_cols = ['Index_Type', 'Precisions', 'p95time', 'RPS']
    
    # 检查必需的列是否存在
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"未找到以下必需的列: {missing_cols}")
    
    # 过滤掉缺失值
    df_clean = df[required_cols].dropna(how='any')
    
    if len(df_clean) == 0:
        print("警告: 删除缺失值后没有有效数据")
        return pd.DataFrame()
    
    # 筛选Precisions大于阈值的配置
    df_high_precision = df_clean[df_clean['Precisions'] > precision_threshold].copy()
    
    print(f"Precisions阈值: {precision_threshold}")
    print(f"Precisions > {precision_threshold} 的配置数量: {len(df_high_precision)}")
    
    if len(df_high_precision) == 0:
        print(f"警告: 没有配置满足 Precisions > {precision_threshold} 的条件")
        return pd.DataFrame()
    
    # 按RPS降序排序，取前N个
    df_sorted = df_high_precision.sort_values('RPS', ascending=False).head(top_n)
    
    print(f"\nRPS排在前{top_n}的配置 (Precisions > {precision_threshold}):")
    print(df_sorted.to_string(index=False))
    
    return df_sorted


def main():
    # 获取数据目录
    script_dir = Path(__file__).parent
    data_dir = Path(DATA_DIR) if DATA_DIR else script_dir
    
    # 构建数据集文件路径
    dataset_path = data_dir / DATASET_FILE
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"文件不存在: {dataset_path}")
    
    print("="*60)
    print(f"分析数据集: {DATASET_FILE}")
    print("="*60)
    
    df = load_dataset(dataset_path)
    
    # 0. 找出Precision大于阈值且RPS排在前10的配置
    top_rps_configs = find_top_rps_configs(df, precision_threshold=PRECISION_THRESHOLD, top_n=10)
    
    # 1. 分析较好的Index Type
    top_index_types = analyze_index_types(df)
    
    # 2. 分析重要旋钮
    important_knobs, feature_importance = analyze_important_knobs(df)
    
    # 3. 分析旋钮范围
    knob_ranges = analyze_knob_ranges(df, important_knobs, precision_threshold=PRECISION_THRESHOLD)
    
    # 汇总结果
    results = {
        'dataset_file': str(dataset_path),
        'top_rps_configs': top_rps_configs.to_dict('records') if not top_rps_configs.empty else [],
        'top_index_types': top_index_types,
        'important_knobs': important_knobs,
        'knob_importance': feature_importance.to_dict('records'),
        'knob_ranges': knob_ranges
    }
    
    # 打印总结
    print("\n" + "="*60)
    print("分析结果总结")
    print("="*60)
    print(f"\n数据集文件: {dataset_path.name}")
    print(f"\n较好的Index Type (前3个): {top_index_types}")
    print(f"\n重要旋钮 (前10个): {important_knobs}")
    print(f"\n旋钮范围已统计，详细信息见上方输出")
    
    # 保存结果到JSON文件
    if OUTPUT_JSON:
        output_path = data_dir / OUTPUT_JSON
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")
    else:
        print("\n提示: 在代码中设置 OUTPUT_JSON 可以保存结果到JSON文件")


if __name__ == "__main__":
    main()
