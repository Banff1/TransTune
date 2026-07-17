#!/usr/bin/env python3
"""
在指定数据集上训练随机森林模型，预测性能指标。
将同一数据集按 80% / 20% 划分为训练集和测试集。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


# ==================== 配置区域 ====================
# 指定数据集文件（相对于 DATA_DIR）
DATASET_FILE = "random-match-int-2048-angular-no-filters.xlsx"

# 训练集比例
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# 随机种子
RANDOM_STATE = 42

# 数据目录
DATA_DIR = "/path/to/TransTune/auto-configure/transtune/perf-predict-model"

# 是否保存模型
SAVE_MODELS = False
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

    possible_name_cols = ['Dataset Name']
    name_col = None
    for col in possible_name_cols:
        if col in df_features.columns:
            name_col = col
            break

    if name_col is None:
        print(
            f"警告: 未找到明确的数据集名称列，"
            f"假设第一列 '{df_features.columns[0]}' 为数据集名称"
        )
        name_col = df_features.columns[0]
        df_features = df_features.rename(columns={name_col: 'dataset_name'})
    elif name_col != 'dataset_name':
        df_features = df_features.rename(columns={name_col: 'dataset_name'})

    return df_features


def load_performance_data(data_dir, file_name):
    """加载指定数据集的性能数据"""
    file_path = Path(data_dir) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    dataset_name = file_path.stem.replace("200-", "")
    print(f"正在加载性能数据: {file_path.name} (数据集: {dataset_name})")

    try:
        df = pd.read_excel(file_path)
    except ImportError as e:
        raise ImportError(
            "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
        ) from e
    except Exception as e:
        raise Exception(f"读取文件 {file_path.name} 时出错: {e}") from e

    print(f"  原始数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")

    df["dataset_name"] = dataset_name

    before_drop = len(df)
    df = df.dropna(how="any")
    after_drop = len(df)
    if after_drop < before_drop:
        print(
            f"  发现缺失值行，已丢弃 {before_drop - after_drop} 行，"
            f"剩余 {after_drop} 行"
        )

    return df, dataset_name


def prepare_features(df_perf, df_features):
    """准备特征：合并数据集特征和系统参数"""
    print("\n正在准备特征...")

    if 'dataset_name' not in df_perf.columns:
        raise ValueError("性能数据中缺少 'dataset_name' 列")
    if 'dataset_name' not in df_features.columns:
        raise ValueError("数据集特征中缺少 'dataset_name' 列")

    df_merged = df_perf.merge(
        df_features,
        on='dataset_name',
        how='left',
        suffixes=('', '_feature')
    )

    if df_merged.shape[0] == 0:
        raise ValueError("合并后数据为空，请检查数据集名称是否匹配")

    unmatched = df_merged[df_merged.isnull().any(axis=1)]['dataset_name'].unique()
    if len(unmatched) > 0:
        print(f"警告: 以下数据集在特征文件中未找到匹配: {unmatched.tolist()}")

    target_columns = ['Precisions', 'p95time', 'RPS']
    available_targets = [col for col in target_columns if col in df_merged.columns]

    if not available_targets:
        raise ValueError(f"未找到目标列。可用列: {df_merged.columns.tolist()}")

    print(f"目标变量: {available_targets}")

    exclude_columns = [
        'dataset_name', 'Dataset Name', 'Iteration', 'Time', 'Time_Step',
        'Time_Total', 'Total Time', 'Mean Time', 'Mean Precisions',
    ] + available_targets
    feature_columns = [col for col in df_merged.columns if col not in exclude_columns]

    print(f"特征列数量: {len(feature_columns)}")
    print(f"特征列: {feature_columns}")

    missing_values = df_merged[feature_columns + available_targets].isnull().sum()
    if missing_values.any():
        print("\n警告: 发现缺失值:")
        print(missing_values[missing_values > 0])
        df_merged = df_merged.dropna(subset=feature_columns + available_targets)
        print(f"删除缺失值后数据形状: {df_merged.shape}")

    print("\n正在处理分类特征...")
    label_encoders = {}
    categorical_columns = []

    for col in feature_columns:
        if df_merged[col].dtype == 'object' or df_merged[col].dtype.name == 'category':
            print(f"  发现分类特征: {col}")
            categorical_columns.append(col)
            le = LabelEncoder()
            df_merged[col] = le.fit_transform(df_merged[col].astype(str))
            label_encoders[col] = le

    if categorical_columns:
        print(f"已编码的分类特征: {categorical_columns}")
    else:
        print("未发现分类特征")

    return df_merged, feature_columns, available_targets, label_encoders


def split_train_test(df_merged, feature_columns, target_columns, test_size, random_state):
    """将数据划分为训练集和测试集（所有目标变量使用相同划分）"""
    print(f"\n正在划分数据集 (训练 {1 - test_size:.0%} / 测试 {test_size:.0%})...")

    indices = np.arange(len(df_merged))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    df_train = df_merged.iloc[train_idx]
    df_test = df_merged.iloc[test_idx]

    print(f"训练集大小: {len(df_train)}, 测试集大小: {len(df_test)}")

    X_train = df_train[feature_columns]
    X_test = df_test[feature_columns]
    y_train_dict = {col: df_train[col].values for col in target_columns}
    y_test_dict = {col: df_test[col].values for col in target_columns}

    return X_train, X_test, y_train_dict, y_test_dict


def train_models(X_train, y_train_dict, random_state=42):
    """训练多个随机森林模型（每个目标变量一个）"""
    print(f"\n正在训练模型...")
    print(f"训练数据形状: {X_train.shape}")

    models = {}
    train_results = {}

    for target_name, y_train in y_train_dict.items():
        print(f"\n--- 训练模型: {target_name} ---")

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = calculate_mape(y_train, y_train_pred)

        print(
            f"训练集 - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, "
            f"MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}"
        )

        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)

        print("\n前10个重要特征:")
        print(feature_importance.head(10).to_string(index=False))

        models[target_name] = model
        train_results[target_name] = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_mape': train_mape,
            'train_r2': train_r2,
            'feature_importance': feature_importance,
        }

    return models, train_results


def calculate_mape(y_true, y_pred):
    """计算 MAPE (Mean Absolute Percentage Error)，单位为 %"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_ranking_loss(y_true, y_pred):
    """计算 ranking loss 及归一化值"""
    n = len(y_true)
    if n < 2:
        return 0, 0, 0.0

    ranking_errors = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                if y_true[i] > y_true[j] and y_pred[i] <= y_pred[j]:
                    ranking_errors += 1
                elif y_true[i] < y_true[j] and y_pred[i] >= y_pred[j]:
                    ranking_errors += 1

    normalized_loss = ranking_errors / total_pairs if total_pairs > 0 else 0.0
    return ranking_errors, total_pairs, normalized_loss


def evaluate_models(models, X_test, y_test_dict):
    """在测试集上评估模型"""
    print(f"\n正在评估模型...")
    print(f"测试数据形状: {X_test.shape}")

    test_results = {}

    for target_name, model in models.items():
        y_test = y_test_dict[target_name]
        y_pred = model.predict(X_test)

        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_mape = calculate_mape(y_test, y_pred)

        ranking_errors, total_pairs, normalized_loss = calculate_ranking_loss(
            y_test, y_pred
        )
        similarity = 1.0 - normalized_loss

        test_results[target_name] = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_r2': test_r2,
            'ranking_errors': ranking_errors,
            'total_pairs': total_pairs,
            'normalized_ranking_loss': normalized_loss,
            'similarity': similarity,
        }

        print(f"\n{target_name}:")
        print(
            f"  测试集 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, "
            f"MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}"
        )
        print(
            f"  Ranking Loss: {ranking_errors}/{total_pairs} = {normalized_loss:.4f}"
        )
        print(f"  相似度 (1 - Ranking Loss): {similarity:.4f}")

    return test_results


def main():
    script_dir = Path(__file__).parent
    data_dir = Path(DATA_DIR) if DATA_DIR else script_dir
    features_file = data_dir / "dataset_features.xlsx"

    if not features_file.exists():
        raise FileNotFoundError(f"数据集特征文件不存在: {features_file}")

    print("=" * 60)
    print(f"数据集文件: {DATASET_FILE}")
    print(f"划分比例: 训练 {TRAIN_RATIO:.0%} / 测试 {TEST_RATIO:.0%}")
    print(f"随机种子: {RANDOM_STATE}")
    print("=" * 60)

    df_features = load_dataset_features(features_file)
    df_perf, dataset_name = load_performance_data(data_dir, DATASET_FILE)

    df_merged, feature_columns, target_columns, label_encoders = prepare_features(
        df_perf, df_features
    )

    X_train, X_test, y_train_dict, y_test_dict = split_train_test(
        df_merged,
        feature_columns,
        target_columns,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)
    models, train_results = train_models(
        X_train, y_train_dict, random_state=RANDOM_STATE
    )

    print("\n" + "=" * 60)
    print("评估测试集")
    print("=" * 60)
    test_results = evaluate_models(models, X_test, y_test_dict)

    print("\n" + "=" * 60)
    print(f"总结 ({dataset_name})")
    print("=" * 60)

    print("\n训练集评估:")
    for target_name, result in train_results.items():
        print(f"\n{target_name}:")
        print(f"  R²: {result['train_r2']:.4f}")
        print(f"  MAE: {result['train_mae']:.4f}")
        print(f"  MAPE: {result['train_mape']:.2f}%")
        print(f"  MSE: {result['train_mse']:.4f}")

    print("\n测试集评估:")
    for target_name, result in test_results.items():
        print(f"\n{target_name}:")
        print(f"  R²: {result['test_r2']:.4f}")
        print(f"  MAE: {result['test_mae']:.4f}")
        print(f"  MAPE: {result['test_mape']:.2f}%")
        print(f"  MSE: {result['test_mse']:.4f}")
        print(f"  Ranking Loss: {result['normalized_ranking_loss']:.4f}")
        print(f"  相似度: {result['similarity']:.4f}")

    if SAVE_MODELS:
        print("\n正在保存模型...")
        for target_name, model in models.items():
            model_file = data_dir / f"rf_model_{target_name}_{dataset_name}.pkl"
            joblib.dump(model, model_file)
            print(f"  已保存: {model_file}")

        feature_info = {
            'feature_columns': feature_columns,
            'target_columns': target_columns,
            'label_encoders': label_encoders,
            'dataset_name': dataset_name,
        }
        feature_info_file = data_dir / f"feature_info_{dataset_name}.pkl"
        joblib.dump(feature_info, feature_info_file)
        print(f"  已保存特征信息: {feature_info_file}")


if __name__ == "__main__":
    main()
