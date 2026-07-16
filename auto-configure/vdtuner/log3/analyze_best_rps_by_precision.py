#!/usr/bin/env python3
"""
读取调优结果 xlsx，在给定 Precisions 阈值下找出 RPS 最高的配置。

用法:
    python find_best_rps_by_precision.py
    python find_best_rps_by_precision.py --file path/to/result.xlsx --thresholds 0.85 0.90 0.95
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_FILE = (
    # "/path/to/TransTune/auto-configure/vdtuner/log3/"
    # "random-100-match-kw-small-vocab-no-filters.xlsx"
    "/path/to/TransTune/auto-configure/vdtuner/log3/arxiv-titles-384-angular-no-filters.xlsx"
)
DEFAULT_THRESHOLDS = [0.85, 0.90, 0.95]

DISPLAY_COLUMNS = [
    "Iteration",
    "Index_Type",
    "nlist",
    "nprobe",
    "m",
    "nbits",
    "M",
    "efConstruction",
    "ef",
    "reorder_k",
    "maxSize",
    "sealProportion",
    "autoHandoff",
    "autoBalance",
    "gracefulTime",
    "insertBufSize",
    "minSegmentSizeToIndex",
    "Precisions",
    "p95time",
    "RPS",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(file_path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "读取 Excel 文件需要 openpyxl 库。请运行: pip install openpyxl"
        ) from exc


def find_best_rps_by_precision(
    df: pd.DataFrame,
    precision_threshold: float,
) -> Optional[pd.Series]:
    """
    在 Precisions > threshold 的配置中，返回 RPS 最高的那一行。
    """
    required_cols = ["Precisions", "RPS"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    df_clean = df[required_cols].dropna(how="any")
    df_filtered = df_clean[df_clean["Precisions"] > precision_threshold]

    if df_filtered.empty:
        return None

    best_idx = df_filtered["RPS"].idxmax()
    return df.loc[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 Precisions 阈值筛选，找出 RPS 最高的配置"
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="输入 xlsx 文件路径",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Precisions 阈值列表，筛选 Precisions > threshold 的配置",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print("=" * 70)
    print(f"数据集: {file_path.name}")
    print("=" * 70)

    df = load_dataset(file_path)
    print(f"总行数: {len(df)}")

    display_cols = [col for col in DISPLAY_COLUMNS if col in df.columns]

    for threshold in args.thresholds:
        print("\n" + "-" * 70)
        print(f"Precisions > {threshold:.2f} 时，RPS 最高的配置")
        print("-" * 70)

        valid_count = df.dropna(subset=["Precisions", "RPS"])
        valid_count = valid_count[valid_count["Precisions"] > threshold]
        print(f"满足条件的配置数量: {len(valid_count)}")

        best_row = find_best_rps_by_precision(df, threshold)
        if best_row is None:
            print(f"警告: 没有配置满足 Precisions > {threshold:.2f}")
            continue

        print(best_row[display_cols].to_string())


if __name__ == "__main__":
    main()
