#!/usr/bin/env python3
"""
训练随机森林模型，预测性能指标
输入：数据集特征 + 系统参数
输出：Precisions, p95time, RPS
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


def load_dataset_features(features_file):
    """加载数据集特征文件"""
    print(f"正在加载数据集特征: {features_file}")
    try:
        df_features = pd.read_excel(features_file)
    except ImportError as e:
        raise ImportError(
            "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
        ) from e
    except Exception as e:
        raise Exception(f"读取数据集特征文件时出错: {e}") from e
    
    print(f"数据集特征形状: {df_features.shape}")
    print(f"数据集特征列: {df_features.columns.tolist()}")
    
    # 检查是否有数据集名称列（可能是 dataset_name, dataset, name 等）
    possible_name_cols = ['Dataset Name']
    name_col = None
    for col in possible_name_cols:
        if col in df_features.columns:
            name_col = col
            break
    
    if name_col is None:
        # 如果没有找到，假设第一列是数据集名称
        print(f"警告: 未找到明确的数据集名称列，假设第一列 '{df_features.columns[0]}' 为数据集名称")
        name_col = df_features.columns[0]
        df_features = df_features.rename(columns={name_col: 'dataset_name'})
    elif name_col != 'dataset_name':
        df_features = df_features.rename(columns={name_col: 'dataset_name'})
    
    return df_features


def load_performance_data(data_dir):
    """加载所有离线测试数据文件"""
    print(f"\n正在加载性能数据文件...")
    all_data = []
    
    # 查找所有 200-*.xlsx 文件
    pattern = "200-*.xlsx"
    data_files = list(Path(data_dir).glob(pattern))
    
    if not data_files:
        raise ValueError(f"在 {data_dir} 中未找到匹配 {pattern} 的文件")
    
    print(f"找到 {len(data_files)} 个性能数据文件")
    
    for file_path in data_files:
        # 从文件名提取数据集名称 (例如: 200-arxiv-titles-384-angular-no-filters.xlsx -> arxiv-titles-384-angular-no-filters)
        dataset_name = file_path.stem.replace("200-", "")
        print(f"  处理文件: {file_path.name} (数据集: {dataset_name})")
        
        try:
            df = pd.read_excel(file_path)
            print(f"    数据形状: {df.shape}")
            print(f"    列名: {df.columns.tolist()}")
            
            # 添加数据集名称列，用于后续合并特征
            df['dataset_name'] = dataset_name
            
            all_data.append(df)
        except ImportError as e:
            raise ImportError(
                "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
            ) from e
        except Exception as e:
            print(f"    警告: 读取文件 {file_path.name} 时出错: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有成功加载任何性能数据文件")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后的性能数据形状: {combined_df.shape}")
    return combined_df


def prepare_features(df_perf, df_features):
    """准备特征：合并数据集特征和系统参数"""
    print("\n正在准备特征...")

    # 检查合并键
    if 'dataset_name' not in df_perf.columns:
        raise ValueError("性能数据中缺少 'dataset_name' 列")
    if 'dataset_name' not in df_features.columns:
        raise ValueError("数据集特征中缺少 'dataset_name' 列")

    # 合并数据集特征
    df_merged = df_perf.merge(
        df_features,
        on='dataset_name',
        how='left',
        suffixes=('', '_feature')
    )

    # 检查合并结果
    if df_merged.shape[0] == 0:
        raise ValueError("合并后数据为空，请检查数据集名称是否匹配")

    unmatched = df_merged[df_merged.isnull().any(axis=1)]['dataset_name'].unique()
    if len(unmatched) > 0:
        print(f"警告: 以下数据集在特征文件中未找到匹配: {unmatched.tolist()}")

    # 识别性能指标列（这些是目标变量）
    target_columns = ['Precisions', 'p95time', 'RPS']
    available_targets = [col for col in target_columns if col in df_merged.columns]

    if not available_targets:
        raise ValueError(f"未找到目标列。可用列: {df_merged.columns.tolist()}")

    print(f"目标变量: {available_targets}")

    # 识别特征列（排除目标变量、数据集名称、索引列、Time列等）
    # Iteration 是实验轮次/索引列，不应作为训练特征
    exclude_columns = ['dataset_name', 'Dataset Name', 'Iteration', 'Time', 'Time_Step', 'Time_Total', 'Total Time', 'Mean Time', 'Mean Precisions'] + available_targets
    feature_columns = [col for col in df_merged.columns if col not in exclude_columns]

    print(f"特征列数量: {len(feature_columns)}")
    # print(f"特征列: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"特征列: {feature_columns}")
    print(f"特征列: {feature_columns}")

    # 检查缺失值
    missing_values = df_merged[feature_columns + available_targets].isnull().sum()
    if missing_values.any():
        print(f"\n警告: 发现缺失值:")
        print(missing_values[missing_values > 0])
        # 删除包含缺失值的行
        df_merged = df_merged.dropna(subset=feature_columns + available_targets)
        print(f"删除缺失值后数据形状: {df_merged.shape}")

    # 处理分类特征
    print("\n正在处理分类特征...")
    label_encoders = {}
    categorical_columns = []

    for col in feature_columns:
        if df_merged[col].dtype == 'object' or df_merged[col].dtype.name == 'category':
            print(f"  发现分类特征: {col}")
            categorical_columns.append(col)

            # 使用LabelEncoder编码分类特征
            le = LabelEncoder()
            df_merged[col] = le.fit_transform(df_merged[col].astype(str))
            label_encoders[col] = le

    if categorical_columns:
        print(f"已编码的分类特征: {categorical_columns}")
    else:
        print("未发现分类特征")

    return df_merged, feature_columns, available_targets, label_encoders


def train_models(X, y_dict, test_size=0.2, random_state=42):
    """训练多个随机森林模型（每个目标变量一个）"""
    print(f"\n正在训练模型...")
    print(f"训练数据形状: {X.shape}")
    
    models = {}
    results = {}
    
    for target_name, y in y_dict.items():
        print(f"\n--- 训练模型: {target_name} ---")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        
        # 训练随机森林模型
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"训练集 - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n前10个重要特征:")
        print(feature_importance.head(10).to_string(index=False))
        
        models[target_name] = model
        results[target_name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    return models, results


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    data_dir = script_dir

    # 文件路径
    features_file = data_dir / "dataset_features.xlsx"

    # 检查文件是否存在
    if not features_file.exists():
        raise FileNotFoundError(f"数据集特征文件不存在: {features_file}")

    # 加载数据
    df_features = load_dataset_features(features_file)
    df_perf = load_performance_data(data_dir)

    # 准备特征
    df_merged, feature_columns, target_columns, label_encoders = prepare_features(df_perf, df_features)

    print(f"特征列: {feature_columns}")
    print(f"目标变量: {target_columns}")

    # 提取特征和目标变量
    X = df_merged[feature_columns]
    y_dict = {col: df_merged[col].values for col in target_columns}

    # 训练模型
    models, results = train_models(X, y_dict)

    # 保存模型
    print(f"\n正在保存模型...")
    for target_name, model in models.items():
        model_file = data_dir / f"rf_model_{target_name}.pkl"
        joblib.dump(model, model_file)
        print(f"  已保存: {model_file}")

    # 保存特征信息和编码器（用于预测时使用）
    feature_info = {
        'feature_columns': feature_columns,
        'target_columns': target_columns,
        'label_encoders': label_encoders
    }
    feature_info_file = data_dir / "feature_info.pkl"
    joblib.dump(feature_info, feature_info_file)
    print(f"  已保存特征信息: {feature_info_file}")

    # 打印总结
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    print(f"\n模型评估总结:")
    for target_name, result in results.items():
        print(f"\n{target_name}:")
        print(f"  R²: {result['test_r2']:.4f}")
        print(f"  MAE: {result['test_mae']:.4f}")
        print(f"  MSE: {result['test_mse']:.4f}")


if __name__ == "__main__":
    main()

