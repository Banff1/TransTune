#!/usr/bin/env python3
"""
训练随机森林模型，预测性能指标，并计算 ranking-loss 作为数据集相似度
输入：训练数据集文件 + 测试数据集文件
输出：训练好的模型 + ranking-loss 相似度
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


# ==================== 配置区域 ====================
# 指定训练文件列表
TRAIN_FILES = [
    # "200-arxiv-titles-384-angular-no-filters.xlsx"
    # "200-deep-image-96-angular.xlsx"
    # "glove-25-angular.xlsx"
    "glove-100-angular.xlsx"
    # "random-match-keyword-100-angular-no-filters.xlsx"
    # "200-random-100-match-kw-small-vocab-no-filters.xlsx"
    # "random-geo-radius-2048-angular-no-filters.xlsx"
    # "random-match-int-2048-angular-no-filters.xlsx"
    # "random-range-2048-angular-no-filters.xlsx"
]

# 指定测试文件
TEST_FILES = [
    # "200-arxiv-titles-384-angular-no-filters.xlsx"
    # "200-deep-image-96-angular.xlsx"
    # "glove-25-angular.xlsx"
    # "glove-100-angular.xlsx"
    # "random-match-keyword-100-angular-no-filters.xlsx"
    # "200-random-100-match-kw-small-vocab-no-filters.xlsx"
    "random-geo-radius-2048-angular-no-filters.xlsx"
    # "random-match-int-2048-angular-no-filters.xlsx"
    # "random-range-2048-angular-no-filters.xlsx"
]

# 从测试文件中随机抽取的样本数量
N_SAMPLES = 30

# 随机种子
RANDOM_STATE = 42

# 数据目录
DATA_DIR = "/path/to/TransTune/auto-configure/vdtuner/perf-predict-model"

# ==================================================


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
    
    # 检查是否有数据集名称列
    possible_name_cols = ['Dataset Name']
    name_col = None
    for col in possible_name_cols:
        if col in df_features.columns:
            name_col = col
            break
    
    if name_col is None:
        print(f"警告: 未找到明确的数据集名称列，假设第一列 '{df_features.columns[0]}' 为数据集名称")
        name_col = df_features.columns[0]
        df_features = df_features.rename(columns={name_col: 'dataset_name'})
    elif name_col != 'dataset_name':
        df_features = df_features.rename(columns={name_col: 'dataset_name'})
    
    return df_features


def load_performance_data_from_files(data_dir, file_names):
    """从指定的文件列表中加载性能数据"""
    print(f"\n正在加载性能数据文件...")
    all_data = []
    
    for file_name in file_names:
        file_path = Path(data_dir) / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 从文件名提取数据集名称
        dataset_name = file_path.stem.replace("200-", "")
        print(f"  处理文件: {file_path.name} (数据集: {dataset_name})")
        
        try:
            df = pd.read_excel(file_path)
            print(f"    原始数据形状: {df.shape}")
            print(f"    列名: {df.columns.tolist()}")
            
            # 先添加数据集名称列
            df["dataset_name"] = dataset_name

            # 丢弃包含任意缺失值的整行，确保这些行既不参与训练也不参与测试/抽样
            before_drop = len(df)
            df = df.dropna(how="any")
            after_drop = len(df)
            if after_drop < before_drop:
                print(
                    f"    发现缺失值行，已丢弃 {before_drop - after_drop} 行，"
                    f"剩余 {after_drop} 行"
                )
            
            all_data.append(df)
        except ImportError as e:
            raise ImportError(
                "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
            ) from e
        except Exception as e:
            raise Exception(f"读取文件 {file_path.name} 时出错: {e}") from e
    
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
    exclude_columns = ['dataset_name', 'Dataset Name', 'Iteration', 'Time', 'Time_Step', 
                      'Time_Total', 'Total Time', 'Mean Time', 'Mean Precisions'] + available_targets
    feature_columns = [col for col in df_merged.columns if col not in exclude_columns]

    print(f"特征列数量: {len(feature_columns)}")
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


def prepare_test_features(df_test_perf, df_features, feature_columns):
    """准备测试特征：合并数据集特征，但不创建新的编码器"""
    print("\n正在准备测试特征...")

    # 检查合并键
    if 'dataset_name' not in df_test_perf.columns:
        raise ValueError("测试性能数据中缺少 'dataset_name' 列")
    if 'dataset_name' not in df_features.columns:
        raise ValueError("数据集特征中缺少 'dataset_name' 列")

    # 合并数据集特征
    df_merged = df_test_perf.merge(
        df_features,
        on='dataset_name',
        how='left',
        suffixes=('', '_feature')
    )

    # 检查合并结果
    if df_merged.shape[0] == 0:
        raise ValueError("合并后测试数据为空，请检查数据集名称是否匹配")

    unmatched = df_merged[df_merged.isnull().any(axis=1)]['dataset_name'].unique()
    if len(unmatched) > 0:
        print(f"警告: 以下数据集在特征文件中未找到匹配: {unmatched.tolist()}")

    # 识别目标列
    target_columns = ['Precisions', 'p95time', 'RPS']
    available_targets = [col for col in target_columns if col in df_merged.columns]

    if not available_targets:
        raise ValueError(f"未找到目标列。可用列: {df_merged.columns.tolist()}")

    print(f"目标变量: {available_targets}")

    # 检查缺失值
    missing_values = df_merged[feature_columns + available_targets].isnull().sum()
    if missing_values.any():
        print(f"\n警告: 发现缺失值:")
        print(missing_values[missing_values > 0])
        # 删除包含缺失值的行
        df_merged = df_merged.dropna(subset=feature_columns + available_targets)
        print(f"删除缺失值后数据形状: {df_merged.shape}")

    return df_merged, available_targets


def encode_test_features(df_test, feature_columns, label_encoders):
    """对测试集的特征进行编码（使用训练集的编码器）"""
    print("\n正在处理测试集分类特征...")
    df_test_encoded = df_test.copy()
    
    for col in feature_columns:
        if col in label_encoders:
            le = label_encoders[col]
            # 处理测试集中可能出现的新类别
            class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
            df_test_encoded[col] = (
                df_test_encoded[col]
                .astype(str)
                .map(class_to_idx)
                .fillna(-1)  # 新类别标记为 -1
                .astype(int)
            )
    
    return df_test_encoded


def train_models(X, y_dict, test_size=0.2, random_state=42):
    """训练多个随机森林模型（每个目标变量一个）"""
    print(f"\n正在训练模型...")
    print(f"训练数据形状: {X.shape}")
    
    models = {}
    results = {}
    
    for target_name, y in y_dict.items():
        print(f"\n--- 训练模型: {target_name} ---")
        
        # 划分训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_valid.shape[0]}")
        
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
        y_valid_pred = model.predict(X_valid)
        
        # 评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        valid_mse = mean_squared_error(y_valid, y_valid_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        valid_mae = mean_absolute_error(y_valid, y_valid_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        valid_r2 = r2_score(y_valid, y_valid_pred)
        
        print(f"训练集 - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"验证集 - MSE: {valid_mse:.4f}, MAE: {valid_mae:.4f}, R²: {valid_r2:.4f}")
        
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
            'valid_mse': valid_mse,
            'train_mae': train_mae,
            'valid_mae': valid_mae,
            'train_r2': train_r2,
            'valid_r2': valid_r2,
            'feature_importance': feature_importance
        }
    
    return models, results


def calculate_ranking_loss(y_true, y_pred):
    """
    计算 ranking loss
    
    Ranking loss 定义为：对于所有样本对 (i, j)，如果 y_true[i] > y_true[j]，
    但 y_pred[i] <= y_pred[j]，则产生损失。
    
    返回：
        ranking_loss: 排序错误的样本对数量
        total_pairs: 总的有效样本对数量
        normalized_loss: 归一化的 ranking loss (0-1之间)
    """
    n = len(y_true)
    if n < 2:
        return 0, 0, 0.0
    
    ranking_errors = 0
    total_pairs = 0
    
    # 遍历所有样本对
    for i in range(n):
        for j in range(i + 1, n):
            # 如果真实值有明确的排序关系
            if y_true[i] != y_true[j]:
                total_pairs += 1
                # 如果真实值 i > j，但预测值 i <= j，则产生错误
                if y_true[i] > y_true[j] and y_pred[i] <= y_pred[j]:
                    ranking_errors += 1
                # 如果真实值 i < j，但预测值 i >= j，则产生错误
                elif y_true[i] < y_true[j] and y_pred[i] >= y_pred[j]:
                    ranking_errors += 1
    
    if total_pairs == 0:
        normalized_loss = 0.0
    else:
        normalized_loss = ranking_errors / total_pairs
    
    return ranking_errors, total_pairs, normalized_loss


def evaluate_with_ranking_loss(models, X_test, y_test_dict, n_samples=None, random_state=42):
    """
    在测试集上评估模型，并计算 ranking loss
    
    Args:
        models: 训练好的模型字典
        X_test: 测试集特征
        y_test_dict: 测试集真实值字典
        n_samples: 随机抽取的样本数量，如果为None则使用全部测试集
        random_state: 随机种子
    
    Returns:
        results: 包含评估指标和 ranking loss 的字典
    """
    print(f"\n正在评估模型...")
    print(f"测试数据形状: {X_test.shape}")
    
    # 随机抽取样本（排除第一行，索引从1开始）
    if n_samples is not None and n_samples < len(X_test):
        np.random.seed(random_state)
        # 排除第一行（索引0），从索引1开始选择
        available_indices = np.arange(1, len(X_test))
        # 确保抽取数量不超过可用样本数
        actual_n_samples = min(n_samples, len(available_indices))
        sample_indices = np.random.choice(available_indices, size=actual_n_samples, replace=False)
        X_test_sample = X_test.iloc[sample_indices].reset_index(drop=True)
        y_test_sample_dict = {k: v[sample_indices] for k, v in y_test_dict.items()}
        print(f"随机抽取了 {actual_n_samples} 个样本进行评估（已排除第一行）")
    else:
        # 即使使用全部数据，也排除第一行
        if len(X_test) > 1:
            X_test_sample = X_test.iloc[1:].reset_index(drop=True)
            y_test_sample_dict = {k: v[1:] for k, v in y_test_dict.items()}
            print(f"使用全部 {len(X_test_sample)} 个样本进行评估（已排除第一行）")
        else:
            X_test_sample = X_test
            y_test_sample_dict = y_test_dict
            print(f"测试数据只有1行，使用全部数据")
    
    results = {}
    
    for target_name, model in models.items():
        y_test = y_test_sample_dict[target_name]
        y_pred = model.predict(X_test_sample)
        
        # 计算传统评估指标
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # 计算 ranking loss
        ranking_errors, total_pairs, normalized_loss = calculate_ranking_loss(y_test, y_pred)
        
        # 相似度 = 1 - normalized_ranking_loss
        similarity = 1.0 - normalized_loss
        
        results[target_name] = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'ranking_errors': ranking_errors,
            'total_pairs': total_pairs,
            'normalized_ranking_loss': normalized_loss,
            'similarity': similarity
        }
        
        print(f"\n{target_name}:")
        print(f"  测试集 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        print(f"  Ranking Loss: {ranking_errors}/{total_pairs} = {normalized_loss:.4f}")
        print(f"  相似度 (1 - Ranking Loss): {similarity:.4f}")
    
    return results


def main():
    # 获取数据目录
    script_dir = Path(__file__).parent
    data_dir = Path(DATA_DIR) if DATA_DIR else script_dir
    
    # 文件路径
    features_file = data_dir / "dataset_features.xlsx"
    
    # 检查文件是否存在
    if not features_file.exists():
        raise FileNotFoundError(f"数据集特征文件不存在: {features_file}")
    
    print("="*60)
    print("训练文件列表:")
    for f in TRAIN_FILES:
        print(f"  - {f}")
    print("\n测试文件列表:")
    for f in TEST_FILES:
        print(f"  - {f}")
    print(f"\n随机抽取样本数: {N_SAMPLES}")
    print("="*60)
    
    # 加载数据集特征
    df_features = load_dataset_features(features_file)
    
    # 加载训练数据
    print("\n" + "="*60)
    print("加载训练数据")
    print("="*60)
    df_train_perf = load_performance_data_from_files(data_dir, TRAIN_FILES)
    
    # 准备训练特征
    df_train_merged, feature_columns, target_columns, label_encoders = prepare_features(
        df_train_perf, df_features
    )
    
    # 提取训练特征和目标变量
    X_train = df_train_merged[feature_columns]
    y_train_dict = {col: df_train_merged[col].values for col in target_columns}
    
    # 训练模型
    print("\n" + "="*60)
    print("训练模型")
    print("="*60)
    models, train_results = train_models(X_train, y_train_dict, random_state=RANDOM_STATE)
    
    # 对每个测试文件进行评估
    print("\n" + "="*60)
    print("评估测试数据")
    print("="*60)
    
    all_test_results = {}
    
    for test_file in TEST_FILES:
        print(f"\n处理测试文件: {test_file}")
        print("-"*60)
        
        # 加载测试数据
        df_test_perf = load_performance_data_from_files(data_dir, [test_file])
        
        # 准备测试特征（合并特征，但不创建新编码器）
        df_test_merged, test_target_columns = prepare_test_features(
            df_test_perf, df_features, feature_columns
        )
        
        # 使用训练集的编码器对测试集进行编码
        df_test_encoded = encode_test_features(df_test_merged, feature_columns, label_encoders)
        
        # 确保测试集包含所有特征列
        for col in feature_columns:
            if col not in df_test_encoded.columns:
                df_test_encoded[col] = 0  # 缺失的特征填充为0
        
        # 删除包含缺失值的行
        df_test_encoded = df_test_encoded.dropna(subset=feature_columns + test_target_columns)
        
        if len(df_test_encoded) == 0:
            print(f"警告: 测试文件 {test_file} 处理后数据为空，跳过")
            continue
        
        # 提取测试特征和目标变量（只使用训练时存在的目标列）
        X_test = df_test_encoded[feature_columns]
        y_test_dict = {col: df_test_encoded[col].values for col in target_columns if col in test_target_columns}
        
        # 评估模型并计算 ranking loss
        test_results = evaluate_with_ranking_loss(
            models, X_test, y_test_dict, n_samples=N_SAMPLES, random_state=RANDOM_STATE
        )
        
        all_test_results[test_file] = test_results
    
    # 打印总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    print("\n训练集评估:")
    for target_name, result in train_results.items():
        print(f"\n{target_name}:")
        print(f"  验证集 R²: {result['valid_r2']:.4f}")
        print(f"  验证集 MAE: {result['valid_mae']:.4f}")
        print(f"  验证集 MSE: {result['valid_mse']:.4f}")
    
    print("\n测试集评估 (Ranking Loss 相似度):")
    for test_file, test_results in all_test_results.items():
        print(f"\n测试文件: {test_file}")
        for target_name, result in test_results.items():
            print(f"  {target_name}:")
            print(f"    R²: {result['test_r2']:.4f}")
            print(f"    MAE: {result['test_mae']:.4f}")
            print(f"    Ranking Loss: {result['normalized_ranking_loss']:.4f}")
            print(f"    相似度: {result['similarity']:.4f}")
    
    # # 保存模型（可选）
    # print(f"\n正在保存模型...")
    # for target_name, model in models.items():
    #     model_file = data_dir / f"rf_model_{target_name}_ranking.pkl"
    #     joblib.dump(model, model_file)
    #     print(f"  已保存: {model_file}")
    
    # # 保存特征信息和编码器
    # feature_info = {
    #     'feature_columns': feature_columns,
    #     'target_columns': target_columns,
    #     'label_encoders': label_encoders
    # }
    # feature_info_file = data_dir / "feature_info_ranking.pkl"
    # joblib.dump(feature_info, feature_info_file)
    # print(f"  已保存特征信息: {feature_info_file}")


if __name__ == "__main__":
    main()

