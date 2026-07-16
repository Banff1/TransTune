#!/usr/bin/env python3
"""
分析数据集，确定：
1. 较好的index type：统计每个type对应的平均p95time，保留3个最好的（不统计缺失值）
2. 重要旋钮：训练随机森林模型预测p95time和Precisions，分别选择重要性排在前10个旋钮
3. 旋钮范围：针对RPS在超出平均RPS一定比例的配置下，统计重要旋钮的取值范围
4. 聚合重要旋钮：将precision和p95time的重要旋钮聚合，按重要性选出前10（排除Index_Type）
5. 聚合旋钮范围：将precision和p95time的旋钮范围取并集

输入：数据集文件xlsx
输出：较好的index type，重要旋钮，旋钮范围，聚合重要旋钮，聚合旋钮范围
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json


# ==================== 配置区域 ====================
# 指定要分析的数据集文件
DATASET_FILE = "random-range-2048-angular-no-filters.xlsx"
# "random-geo-radius-2048-angular-no-filters.xlsx"

# 输出JSON文件路径（如果为None则不保存）
OUTPUT_JSON = None  #"random-geo-radius-2048-angular-no-filters.json"  
# 例如: "analysis_results.json"

# Precisions阈值（用于分析旋钮范围时的过滤条件）
PRECISION_THRESHOLD = 0.9  # 只统计Precisions大于此值的配置

#高于平均RPS比例, 在此之上的才作为旋钮范围的依据
RPS_THRESHOLD = 1.1

# 数据目录
DATA_DIR = "/path/to/TransTune/auto-configure/vdtuner/log3"
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
    训练随机森林模型预测p95time和Precisions，分别选择重要性排在前10个旋钮
    
    Returns:
        tuple: (p95time_top_10_knobs, precision_top_10_knobs, p95time_feature_importance, precision_feature_importance)
    """
    print("\n" + "="*60)
    print("2. 分析重要旋钮")
    print("="*60)
    
    if 'p95time' not in df.columns:
        raise ValueError("未找到 'p95time' 列")
    
    if 'Precisions' not in df.columns:
        raise ValueError("未找到 'Precisions' 列")
    
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
    feature_cols = available_knobs + ['p95time', 'Precisions']
    before_drop = len(df)
    df_clean = df[feature_cols].dropna(how='any')
    after_drop = len(df_clean)
    
    if after_drop == 0:
        raise ValueError("删除缺失值后没有有效数据")
    
    print(f"原始数据行数: {before_drop}")
    print(f"删除包含缺失值的行后，有效数据行数: {after_drop}")
    if before_drop > after_drop:
        print(f"已删除 {before_drop - after_drop} 行包含缺失值的数据")
    
    # 分离特征和目标
    X = df_clean[available_knobs].copy()
    y_p95time = df_clean['p95time'].values
    y_precision = df_clean['Precisions'].values
    
    # 处理分类特征
    label_encoders = {}
    for col in available_knobs:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # 训练p95time预测模型
    print("\n" + "-"*60)
    print("训练p95time预测模型...")
    print("-"*60)
    model_p95time = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model_p95time.fit(X, y_p95time)
    
    # 获取p95time模型的特征重要性
    p95time_feature_importance = pd.DataFrame({
        'knob': available_knobs,
        'importance': model_p95time.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n所有旋钮的重要性排序 (p95time模型):")
    print(p95time_feature_importance.to_string(index=False))
    
    # 选择p95time模型的前10个重要旋钮
    p95time_top_10_knobs = p95time_feature_importance.head(10)['knob'].tolist()
    
    print(f"\np95time模型的前10个重要旋钮:")
    print(p95time_feature_importance.head(10).to_string(index=False))
    
    # 训练Precisions预测模型
    print("\n" + "-"*60)
    print("训练Precisions预测模型...")
    print("-"*60)
    model_precision = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model_precision.fit(X, y_precision)
    
    # 获取Precisions模型的特征重要性
    precision_feature_importance = pd.DataFrame({
        'knob': available_knobs,
        'importance': model_precision.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n所有旋钮的重要性排序 (Precisions模型):")
    print(precision_feature_importance.to_string(index=False))
    
    # 选择Precisions模型的前10个重要旋钮
    precision_top_10_knobs = precision_feature_importance.head(10)['knob'].tolist()
    
    print(f"\nPrecisions模型的前10个重要旋钮:")
    print(precision_feature_importance.head(10).to_string(index=False))
    
    return (p95time_top_10_knobs, precision_top_10_knobs, 
            p95time_feature_importance, precision_feature_importance)


def analyze_knob_ranges(df, p95time_knobs, precision_knobs, precision_threshold=0.80, rps_threshold=1.3):
    """
    分析旋钮范围
    1. 首先过滤出Precisions大于阈值的配置
    2. 在这些配置中计算平均RPS
    3. 找出RPS超出平均RPS一定比例的配置
    4. 统计这些配置中重要旋钮的取值范围（分别统计p95time和precision模型的重要旋钮）
    
    Args:
        df: 数据框
        p95time_knobs: p95time模型的重要旋钮列表
        precision_knobs: precision模型的重要旋钮列表
        precision_threshold: Precisions阈值，默认0.80
        rps_threshold: RPS阈值倍数，默认1.3（即平均RPS的1.3倍）
    """
    print("\n" + "="*60)
    print("3. 分析重要旋钮的取值范围")
    print("="*60)
    
    if 'RPS' not in df.columns:
        raise ValueError("未找到 'RPS' 列")
    
    if 'Precisions' not in df.columns:
        raise ValueError("未找到 'Precisions' 列")
    
    # 合并所有重要旋钮（去重）
    all_important_knobs = list(set(p95time_knobs + precision_knobs))
    
    # 过滤掉缺失值（如果一行中任何列有缺失值，则删除该行）
    before_drop = len(df)
    required_cols = all_important_knobs + ['RPS', 'Precisions']
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
        return {'p95time_knobs': {}, 'precision_knobs': {}}
    
    # 第二步：在这些配置中计算平均RPS
    avg_rps = df_high_precision['RPS'].mean()
    print(f"Precisions > {precision_threshold} 的配置中，平均RPS: {avg_rps:.2f}")
    
    # 第三步：筛选RPS超出平均RPS一定比例的配置
    threshold_rps = avg_rps * rps_threshold
    df_filtered = df_high_precision[df_high_precision['RPS'] >= threshold_rps].copy()
    
    print(f"RPS阈值 (平均RPS * {rps_threshold}): {threshold_rps:.2f}")
    print(f"满足条件的配置数量 (Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f}): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print(f"警告: 没有配置满足 Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f} 的条件")
        return {'p95time_knobs': {}, 'precision_knobs': {}}
    
    # 辅助函数：统计旋钮范围
    def calculate_knob_ranges(knobs_list, df_data, model_name):
        ranges = {}
        for knob in knobs_list:
            if knob not in df_data.columns:
                continue
            
            knob_values = df_data[knob]
            
            # 如果是数值型
            if pd.api.types.is_numeric_dtype(knob_values):
                ranges[knob] = {
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
                ranges[knob] = {
                    'type': 'categorical',
                    'unique_values': sorted([str(v) for v in unique_values]),
                    'value_counts': knob_values.value_counts().to_dict()
                }
        return ranges
    
    # 第四步：分别统计p95time和precision模型的重要旋钮取值范围
    p95time_ranges = calculate_knob_ranges(p95time_knobs, df_filtered, 'p95time')
    precision_ranges = calculate_knob_ranges(precision_knobs, df_filtered, 'precision')
    
    # 打印p95time模型的重要旋钮范围
    print(f"\n" + "-"*60)
    print(f"p95time模型的重要旋钮取值范围 (Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f}):")
    print("-"*60)
    for knob, range_info in p95time_ranges.items():
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
    
    # 打印precision模型的重要旋钮范围
    print(f"\n" + "-"*60)
    print(f"Precisions模型的重要旋钮取值范围 (Precisions > {precision_threshold} 且 RPS >= {threshold_rps:.2f}):")
    print("-"*60)
    for knob, range_info in precision_ranges.items():
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
    
    return {
        'p95time_knobs': p95time_ranges,
        'precision_knobs': precision_ranges
    }


def aggregate_important_knobs(p95time_feature_importance, precision_feature_importance, top_n=10, exclude_knobs=None):
    """
    聚合precision和p95time的重要旋钮，根据重要性选出排名前N的旋钮（排除指定旋钮如Index_Type）
    
    Args:
        p95time_feature_importance: p95time模型的特征重要性DataFrame (knob, importance)
        precision_feature_importance: precision模型的特征重要性DataFrame (knob, importance)
        top_n: 选出前N个旋钮，默认10
        exclude_knobs: 要排除的旋钮列表，默认['Index_Type']
    
    Returns:
        list: 聚合后的前N个重要旋钮
        pd.DataFrame: 聚合重要性排序表
    """
    if exclude_knobs is None:
        exclude_knobs = ['Index_Type']
    
    # 构建knob -> (p95time_importance, precision_importance) 的映射
    p95time_dict = dict(zip(p95time_feature_importance['knob'], p95time_feature_importance['importance']))
    precision_dict = dict(zip(precision_feature_importance['knob'], precision_feature_importance['importance']))
    
    all_knobs = set(p95time_dict.keys()) | set(precision_dict.keys())
    
    # 计算聚合重要性（取两者之和，若某模型无该旋钮则记为0）
    aggregated = []
    for knob in all_knobs:
        if knob in exclude_knobs:
            continue
        imp_p95 = p95time_dict.get(knob, 0.0)
        imp_prec = precision_dict.get(knob, 0.0)
        aggregated.append({
            'knob': knob,
            'p95time_importance': imp_p95,
            'precision_importance': imp_prec,
            'aggregated_importance': imp_p95 + imp_prec
        })
    
    agg_df = pd.DataFrame(aggregated).sort_values('aggregated_importance', ascending=False)
    top_knobs = agg_df.head(top_n)['knob'].tolist()
    
    return top_knobs, agg_df


def aggregate_knob_ranges_union(p95time_ranges, precision_ranges):
    """
    聚合precision和p95time的旋钮范围，取并集
    
    对于数值型：min取两者较小值，max取两者较大值
    对于分类型：unique_values取并集
    
    Args:
        p95time_ranges: p95time模型的重要旋钮范围字典
        precision_ranges: precision模型的重要旋钮范围字典
    
    Returns:
        dict: 聚合后的旋钮范围（并集）
    """
    all_knobs = set(p95time_ranges.keys()) | set(precision_ranges.keys())
    aggregated_ranges = {}
    
    for knob in all_knobs:
        r1 = p95time_ranges.get(knob)
        r2 = precision_ranges.get(knob)
        
        if r1 is None:
            aggregated_ranges[knob] = r2.copy()
            continue
        if r2 is None:
            aggregated_ranges[knob] = r1.copy()
            continue
        
        if r1['type'] == 'numeric' and r2['type'] == 'numeric':
            all_vals = r1.get('unique_values', []) + r2.get('unique_values', [])
            aggregated_ranges[knob] = {
                'type': 'numeric',
                'min': min(r1['min'], r2['min']),
                'max': max(r1['max'], r2['max']),
                'mean': (r1['mean'] + r2['mean']) / 2,
                'median': (r1['median'] + r2['median']) / 2,
                'unique_values': sorted(set(all_vals))
            }
        elif r1['type'] == 'categorical' and r2['type'] == 'categorical':
            uv1 = set(str(v) for v in r1['unique_values'])
            uv2 = set(str(v) for v in r2['unique_values'])
            merged_values = sorted(uv1 | uv2)
            # 合并value_counts
            vc1 = r1.get('value_counts', {})
            vc2 = r2.get('value_counts', {})
            merged_counts = {}
            for k in set(vc1.keys()) | set(vc2.keys()):
                merged_counts[str(k)] = vc1.get(k, 0) + vc2.get(k, 0)
            aggregated_ranges[knob] = {
                'type': 'categorical',
                'unique_values': merged_values,
                'value_counts': merged_counts
            }
        else:
            # 类型不一致时，取信息更全的
            aggregated_ranges[knob] = r1 if len(r1.get('unique_values', [])) >= len(r2.get('unique_values', [])) else r2
            aggregated_ranges[knob] = aggregated_ranges[knob].copy()
    
    return aggregated_ranges


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
    
    # 2. 分析重要旋钮（p95time和precision两个模型）
    (p95time_top_10_knobs, precision_top_10_knobs, 
     p95time_feature_importance, precision_feature_importance) = analyze_important_knobs(df)
    
    # 3. 分析旋钮范围（分别分析p95time和precision模型的重要旋钮）
    knob_ranges = analyze_knob_ranges(
        df, 
        p95time_top_10_knobs, 
        precision_top_10_knobs, 
        precision_threshold=PRECISION_THRESHOLD,
        rps_threshold=RPS_THRESHOLD
    )
    
    # 4. 聚合重要旋钮：合并precision和p95time，按重要性选前10（排除Index_Type）
    aggregated_top_knobs, aggregated_importance_df = aggregate_important_knobs(
        p95time_feature_importance, 
        precision_feature_importance, 
        top_n=10, 
        exclude_knobs=['Index_Type']
    )
    
    print("\n" + "="*60)
    print("4. 聚合重要旋钮 (precision + p95time, 排除Index_Type, 前10)")
    print("="*60)
    print(f"\n聚合重要性排序 (p95time_importance + precision_importance):")
    print(aggregated_importance_df.to_string(index=False))
    print(f"\n聚合后的前10个重要旋钮: {aggregated_top_knobs}")
    
    # 5. 聚合旋钮范围：取precision和p95time范围的并集
    aggregated_ranges = aggregate_knob_ranges_union(
        knob_ranges['p95time_knobs'], 
        knob_ranges['precision_knobs']
    )
    
    # 只保留聚合后前10重要旋钮的范围
    aggregated_top_knob_ranges = {k: v for k, v in aggregated_ranges.items() if k in aggregated_top_knobs}
    
    print("\n" + "="*60)
    print("5. 聚合旋钮范围 (precision与p95time范围取并集)")
    print("="*60)
    for knob, range_info in aggregated_top_knob_ranges.items():
        print(f"\n{knob}:")
        if range_info['type'] == 'numeric':
            print(f"  类型: 数值型")
            print(f"  最小值: {range_info['min']}")
            print(f"  最大值: {range_info['max']}")
            print(f"  唯一值数量: {len(range_info['unique_values'])}")
            if len(range_info['unique_values']) <= 20:
                print(f"  唯一值: {range_info['unique_values']}")
        else:
            print(f"  类型: 分类型")
            print(f"  唯一值: {range_info['unique_values']}")
    
    # 汇总结果
    results = {
        'dataset_file': str(dataset_path),
        'top_rps_configs': top_rps_configs.to_dict('records') if not top_rps_configs.empty else [],
        'top_index_types': top_index_types,
        'p95time_important_knobs': p95time_top_10_knobs,
        'precision_important_knobs': precision_top_10_knobs,
        'aggregated_important_knobs': aggregated_top_knobs,
        'aggregated_knob_importance': aggregated_importance_df.to_dict('records'),
        'p95time_knob_importance': p95time_feature_importance.to_dict('records'),
        'precision_knob_importance': precision_feature_importance.to_dict('records'),
        'knob_ranges': knob_ranges,
        'aggregated_knob_ranges': aggregated_top_knob_ranges
    }
    
    # 打印总结
    print("\n" + "="*60)
    print("分析结果总结")
    print("="*60)
    print(f"\n数据集文件: {dataset_path.name}")
    print(f"\n较好的Index Type (前3个): {top_index_types}")
    print(f"\np95time模型的重要旋钮 (前10个): {p95time_top_10_knobs}")
    print(f"\nPrecisions模型的重要旋钮 (前10个): {precision_top_10_knobs}")
    print(f"\n聚合重要旋钮 (precision+p95time, 排除Index_Type, 前10): {aggregated_top_knobs}")
    print(f"\n旋钮范围已统计，聚合范围(并集)见上方输出")
    
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
