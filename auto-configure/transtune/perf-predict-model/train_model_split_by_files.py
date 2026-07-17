#!/usr/bin/env python3
"""
基于数据集特征计算相似度，选取相似数据集训练模型，
并对测试数据集进行性能预测。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset_features(features_file: Path) -> pd.DataFrame:
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

    possible_name_cols = ["Dataset Name"]
    name_col = None
    for col in possible_name_cols:
        if col in df_features.columns:
            name_col = col
            break

    if name_col is None:
        print(
            f"警告: 未找到明确的数据集名称列，假设第一列 '{df_features.columns[0]}' 为数据集名称"
        )
        name_col = df_features.columns[0]
        df_features = df_features.rename(columns={name_col: "dataset_name"})
    elif name_col != "dataset_name":
        df_features = df_features.rename(columns={name_col: "dataset_name"})

    return df_features


def find_performance_files(data_dir: Path, pattern: str = "200-*.xlsx") -> list[Path]:
    files = list(data_dir.glob(pattern))
    if not files:
        raise ValueError(f"在 {data_dir} 中未找到匹配 {pattern} 的文件")
    return sorted(files)


def resolve_train_files(train_names: list[str], all_files: list[Path]) -> list[Path]:
    index = {}
    for path in all_files:
        dataset_name = path.stem.replace("200-", "", 1)
        for key in {path.name, path.stem, dataset_name}:
            index[key.lower()] = path

    resolved = []
    missing = []
    for name in train_names:
        key = name.lower()
        if key not in index and not key.endswith(".xlsx"):
            key = f"{key}.xlsx"
        path = index.get(key)
        if path is None:
            missing.append(name)
        else:
            resolved.append(path)

    if missing:
        available = ", ".join(sorted({p.name for p in all_files}))
        raise ValueError(
            f"未找到以下训练文件: {missing}. 可用文件: {available}"
        )

    # 去重但保持顺序
    seen = set()
    unique_paths = []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def load_performance_data_from_files(files: list[Path]) -> pd.DataFrame:
    """加载指定的性能数据文件"""
    all_data = []
    for file_path in files:
        dataset_name = file_path.stem.replace("200-", "")
        print(f"  处理文件: {file_path.name} (数据集: {dataset_name})")
        try:
            df = pd.read_excel(file_path)
            print(f"    数据形状: {df.shape}")
            print(f"    列名: {df.columns.tolist()}")
            df["dataset_name"] = dataset_name
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
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"合并后的性能数据形状: {combined_df.shape}")
    return combined_df


def merge_perf_features(df_perf: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    if "dataset_name" not in df_perf.columns:
        raise ValueError("性能数据中缺少 'dataset_name' 列")
    if "dataset_name" not in df_features.columns:
        raise ValueError("数据集特征中缺少 'dataset_name' 列")

    df_merged = df_perf.merge(
        df_features,
        on="dataset_name",
        how="left",
        suffixes=("", "_feature"),
    )
    if df_merged.shape[0] == 0:
        raise ValueError("合并后数据为空，请检查数据集名称是否匹配")

    unmatched = df_merged[df_merged.isnull().any(axis=1)]["dataset_name"].unique()
    if len(unmatched) > 0:
        print(f"警告: 以下数据集在特征文件中未找到匹配: {unmatched.tolist()}")
    return df_merged


def infer_columns(df_merged: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_columns = ["Precisions", "p95time", "RPS"]
    available_targets = [col for col in target_columns if col in df_merged.columns]
    if not available_targets:
        raise ValueError(f"未找到目标列。可用列: {df_merged.columns.tolist()}")

    exclude_columns = [
        "dataset_name",
        "Dataset Name",
        "Iteration",
        "Time",
        "Time_Step",
        "Time_Total",
        "Total Time",
        "Mean Time",
        "Mean Precisions",
    ] + available_targets
    feature_columns = [col for col in df_merged.columns if col not in exclude_columns]

    print(f"目标变量: {available_targets}")
    print(f"特征列数量: {len(feature_columns)}")
    print(f"特征列: {feature_columns}")
    return feature_columns, available_targets


def infer_dataset_feature_columns(
    df_features: pd.DataFrame, exclude_columns: list[str]
) -> list[str]:
    columns = [col for col in df_features.columns if col not in exclude_columns]
    if not columns:
        raise ValueError("数据集特征列为空，请检查 exclude_columns 设置")
    return columns


def build_dataset_feature_matrix(
    df_features: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    df = df_features.copy()
    for col in feature_columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df_feature_values = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    for col in df_feature_values.columns:
        if df_feature_values[col].isnull().any():
            mean_value = df_feature_values[col].mean()
            if np.isnan(mean_value):
                mean_value = 0.0
            df_feature_values[col] = df_feature_values[col].fillna(mean_value)

    for col in df_feature_values.columns:
        col_mean = df_feature_values[col].mean()
        col_std = df_feature_values[col].std(ddof=0)
        if col_std == 0 or np.isnan(col_std):
            df_feature_values[col] = 0.0
        else:
            df_feature_values[col] = (df_feature_values[col] - col_mean) / col_std

    return df["dataset_name"], df_feature_values


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    denom = x_norm @ y_norm.T
    denom[denom == 0] = 1.0
    return (x @ y.T) / denom


def select_similar_datasets(
    test_dataset: str,
    candidate_datasets: list[str],
    dataset_names: pd.Series,
    feature_matrix: pd.DataFrame,
    top_k: int,
) -> list[str]:
    dataset_to_idx = {name: idx for idx, name in enumerate(dataset_names.tolist())}
    if test_dataset not in dataset_to_idx:
        raise ValueError(f"测试数据集 {test_dataset} 不在特征文件中")

    test_idx = dataset_to_idx[test_dataset]
    candidate_indices = []
    candidate_names = []
    for name in candidate_datasets:
        if name not in dataset_to_idx:
            print(f"警告: 候选数据集 {name} 不在特征文件中，已跳过")
            continue
        candidate_indices.append(dataset_to_idx[name])
        candidate_names.append(name)

    if not candidate_indices:
        raise ValueError("候选数据集为空，无法进行相似度选择")

    test_vec = feature_matrix.iloc[[test_idx]].to_numpy()
    cand_vecs = feature_matrix.iloc[candidate_indices].to_numpy()
    sims = cosine_similarity_matrix(test_vec, cand_vecs).flatten()

    sorted_idx = np.argsort(sims)[::-1]
    if top_k > 0:
        sorted_idx = sorted_idx[: min(top_k, len(sorted_idx))]
    selected = [candidate_names[i] for i in sorted_idx]

    print(f"\n与测试数据集 {test_dataset} 最相似的训练数据集:")
    for rank, i in enumerate(sorted_idx, 1):
        print(f"  {rank}. {candidate_names[i]} (相似度: {sims[i]:.4f})")

    return selected


def drop_missing(df_merged: pd.DataFrame, feature_columns: list[str], target_columns: list[str]) -> pd.DataFrame:
    missing_values = df_merged[feature_columns + target_columns].isnull().sum()
    if missing_values.any():
        print("\n警告: 发现缺失值:")
        print(missing_values[missing_values > 0])
        df_merged = df_merged.dropna(subset=feature_columns + target_columns)
        print(f"删除缺失值后数据形状: {df_merged.shape}")
    return df_merged


def encode_categoricals_train(
    df_merged: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], list[str]]:
    print("\n正在处理训练集分类特征...")
    label_encoders = {}
    categorical_columns = []

    for col in feature_columns:
        if df_merged[col].dtype == "object" or df_merged[col].dtype.name == "category":
            print(f"  发现分类特征: {col}")
            categorical_columns.append(col)
            le = LabelEncoder()
            df_merged[col] = le.fit_transform(df_merged[col].astype(str))
            label_encoders[col] = le

    if categorical_columns:
        print(f"已编码的分类特征: {categorical_columns}")
    else:
        print("未发现分类特征")

    return df_merged, label_encoders, categorical_columns


def encode_categoricals_test(
    df_merged: pd.DataFrame, feature_columns: list[str], label_encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    print("\n正在处理测试集分类特征...")
    for col in feature_columns:
        if col in label_encoders:
            le = label_encoders[col]
            class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
            df_merged[col] = (
                df_merged[col]
                .astype(str)
                .map(class_to_idx)
                .fillna(-1)
                .astype(int)
            )
    return df_merged


def align_feature_columns(df_merged: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in feature_columns if col not in df_merged.columns]
    if missing_cols:
        print(f"警告: 测试集中缺少特征列: {missing_cols}")
        for col in missing_cols:
            df_merged[col] = np.nan
    return df_merged


def train_models(
    X: pd.DataFrame, y_dict: dict[str, np.ndarray], test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[str, RandomForestRegressor], dict[str, dict]]:
    print("\n正在训练模型...")
    print(f"训练数据形状: {X.shape}")
    models = {}
    results = {}

    for target_name, y in y_dict.items():
        print(f"\n--- 训练模型: {target_name} ---")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_valid.shape[0]}")

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
        y_valid_pred = model.predict(X_valid)

        train_mse = mean_squared_error(y_train, y_train_pred)
        valid_mse = mean_squared_error(y_valid, y_valid_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        valid_mae = mean_absolute_error(y_valid, y_valid_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        valid_r2 = r2_score(y_valid, y_valid_pred)

        print(f"训练集 - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"验证集 - MSE: {valid_mse:.4f}, MAE: {valid_mae:.4f}, R²: {valid_r2:.4f}")

        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("\n前10个重要特征:")
        print(feature_importance.head(10).to_string(index=False))

        models[target_name] = model
        results[target_name] = {
            "train_mse": train_mse,
            "valid_mse": valid_mse,
            "train_mae": train_mae,
            "valid_mae": valid_mae,
            "train_r2": train_r2,
            "valid_r2": valid_r2,
            "feature_importance": feature_importance,
        }

    return models, results


def evaluate_on_test(
    models: dict[str, RandomForestRegressor],
    X_test: pd.DataFrame,
    y_test_dict: dict[str, np.ndarray],
) -> dict[str, dict]:
    print("\n正在测试模型...")
    print(f"测试数据形状: {X_test.shape}")
    results = {}
    for target_name, model in models.items():
        y_test = y_test_dict[target_name]
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        results[target_name] = {
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }
        print(f"\n{target_name}:")
        print(f"  测试集 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    return results


DATA_DIR = "/path/to/TransTune/auto-configure/transtune/perf-predict-model"
FEATURES_FILE = "/path/to/TransTune/auto-configure/transtune/perf-predict-model/dataset_features.xlsx"
# TRAIN_FILES = [
    # "200-arxiv-titles-384-angular-no-filters.xlsx",
    # "200-deep-image-96-angular.xlsx",
    # "200-random_match_keyword.xlsx",
    # "200-random-100-match-kw-small-vocab-no-filters.xlsx",
    # # "200-random-geo-radius-2048-angular-no-filters.xlsx"
    # "200-random-match-int-2048-angular-no-filters.xlsx"
    # # "200-random-range-2048-angular-no-filters.xlsx"
    # ]
TEST_FILES = ["200-glove-25-angular.xlsx"]
TOP_K_SIMILAR = 3
SAVE_MODELS = False

# 使用测试数据集前 N 行加入训练集，其余用于测试
USE_TEST_PREFIX_FOR_TRAIN = False
TEST_PREFIX_TRAIN_ROWS = 20

DATASET_EXCLUDE_COLUMNS = [
    "dataset_name",
    "Dataset Name",
    "Iteration",
    "Time",
    "Time_Step",
    "Time_Total",
    "Total Time",
    "Mean Time",
    "Mean Precisions",
    "Precisions",
    "p95time",
    "RPS",
]

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = Path(DATA_DIR) if DATA_DIR else script_dir
    features_file = Path(FEATURES_FILE) if FEATURES_FILE else data_dir / "dataset_features.xlsx"

    if not features_file.exists():
        raise FileNotFoundError(f"数据集特征文件不存在: {features_file}")

    all_files = find_performance_files(data_dir)
    test_files = resolve_train_files(TEST_FILES, all_files) if TEST_FILES else all_files

    df_features = load_dataset_features(features_file)
    dataset_feature_columns = infer_dataset_feature_columns(
        df_features, DATASET_EXCLUDE_COLUMNS
    )
    dataset_names, dataset_feature_matrix = build_dataset_feature_matrix(
        df_features, dataset_feature_columns
    )

    for test_file in test_files:
        test_dataset_name = test_file.stem.replace("200-", "", 1)
        candidate_files = [p for p in all_files if p != test_file]
        candidate_names = [p.stem.replace("200-", "", 1) for p in candidate_files]

        selected_train_names = select_similar_datasets(
            test_dataset_name,
            candidate_names,
            dataset_names,
            dataset_feature_matrix,
            TOP_K_SIMILAR,
        )
        selected_train_files = [
            p for p in candidate_files if p.stem.replace("200-", "", 1) in set(selected_train_names)
        ]

        # print(selected_train_files)
        # selected_train_files = [Path('/path/to/TransTune/auto-configure/transtune/perf-predict-model/200-glove-100-angular.xlsx')]

        print("\n训练文件:")
        for path in selected_train_files:
            print(f"  - {path.name}")
        print("\n测试文件:")
        print(f"  - {test_file.name}")

        print("\n正在加载训练数据...")
        df_train_perf = load_performance_data_from_files(selected_train_files)
        print("\n正在加载测试数据...")
        df_test_perf = load_performance_data_from_files([test_file])

        if USE_TEST_PREFIX_FOR_TRAIN and TEST_PREFIX_TRAIN_ROWS > 0:
            prefix_rows = min(TEST_PREFIX_TRAIN_ROWS, len(df_test_perf))
            if prefix_rows > 0:
                print(
                    f"\n使用测试数据集前 {prefix_rows} 行加入训练集，其余用于测试"
                )
                df_test_prefix = df_test_perf.iloc[:prefix_rows].copy()
                df_test_perf = df_test_perf.iloc[prefix_rows:].copy()
                df_train_perf = pd.concat(
                    [df_train_perf, df_test_prefix], ignore_index=True
                )
            else:
                print("\n测试数据集为空，跳过前 N 行训练")

        df_train = merge_perf_features(df_train_perf, df_features)
        feature_columns, target_columns = infer_columns(df_train)
        df_train = drop_missing(df_train, feature_columns, target_columns)
        df_train, label_encoders, _ = encode_categoricals_train(df_train, feature_columns)

        if df_test_perf.empty:
            print("\n测试数据为空，跳过测试评估")
            continue

        df_test = merge_perf_features(df_test_perf, df_features)
        df_test = align_feature_columns(df_test, feature_columns)
        df_test = drop_missing(df_test, feature_columns, target_columns)
        df_test = encode_categoricals_test(df_test, feature_columns, label_encoders)

        X_train = df_train[feature_columns]
        y_train_dict = {col: df_train[col].values for col in target_columns}
        X_test = df_test[feature_columns]
        y_test_dict = {col: df_test[col].values for col in target_columns}

        models, _ = train_models(X_train, y_train_dict)
        test_results = evaluate_on_test(models, X_test, y_test_dict)

        if SAVE_MODELS:
            print("\n正在保存模型...")
            for target_name, model in models.items():
                model_file = data_dir / f"rf_model_{target_name}_{test_dataset_name}.pkl"
                joblib.dump(model, model_file)
                print(f"  已保存: {model_file}")

        print(f"\n{'='*60}")
        print(f"{test_dataset_name} 训练/测试完成！")
        print(f"{'='*60}")
        print("\n测试集评估总结:")
        for target_name, result in test_results.items():
            print(f"\n{target_name}:")
            print(f"  R²: {result['test_r2']:.4f}")
            print(f"  MAE: {result['test_mae']:.4f}")
            print(f"  MSE: {result['test_mse']:.4f}")



