#!/usr/bin/env python3
"""
读取调优结果 xlsx，在给定 Precisions 阈值和 Time_Total 预算下找出 RPS 最高的配置。

对每个 Time_Total 预算，筛选:
    Time_Total < budget 且 Precisions > precision_threshold
然后返回 RPS 最高的那一行。

用法:
    python analyze_best_rps_by_time_budget.py
    python analyze_best_rps_by_time_budget.py --file path/to/result.xlsx
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ==================== 配置区域 ====================
DEFAULT_FILE = (
    "/path/to/TransTune/auto-configure/vdtuner/prior/random-100-match-kw-small-vocab-no-filters-new-high.xlsx"
)
PRECISION_THRESHOLD = 0.95  # 只统计 Precisions 大于此值的配置
DEFAULT_TIME_BUDGETS = [3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800]
# ==================================================

DISPLAY_COLUMNS = [
    "Iteration",
    "Time_Total",
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


def find_best_rps_by_time_budget(
    df: pd.DataFrame,
    precision_threshold: float,
    time_budget: float,
) -> Optional[pd.Series]:
    """
    在 Time_Total < time_budget 且 Precisions > precision_threshold 的配置中，
    返回 RPS 最高的那一行。
    """
    required_cols = ["Time_Total", "Precisions", "RPS"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    df_clean = df[required_cols].dropna(how="any")
    df_filtered = df_clean[
        (df_clean["Time_Total"] < time_budget)
        & (df_clean["Precisions"] > precision_threshold)
    ]

    if df_filtered.empty:
        return None

    best_idx = df_filtered["RPS"].idxmax()
    return df.loc[best_idx]


def count_matching_configs(
    df: pd.DataFrame,
    precision_threshold: float,
    time_budget: float,
) -> int:
    df_valid = df.dropna(subset=["Time_Total", "Precisions", "RPS"])
    return len(
        df_valid[
            (df_valid["Time_Total"] < time_budget)
            & (df_valid["Precisions"] > precision_threshold)
        ]
    )


def build_summary_table(
    df: pd.DataFrame,
    precision_threshold: float,
    time_budgets: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for time_budget in time_budgets:
        match_count = count_matching_configs(df, precision_threshold, time_budget)
        best_row = find_best_rps_by_time_budget(df, precision_threshold, time_budget)

        if best_row is None:
            rows.append(
                {
                    "Time_Total 预算": f"< {time_budget:.0f}",
                    "满足条件数": match_count,
                    "最高 RPS": None,
                    "Index_Type": None,
                    "Time_Total": None,
                    "Precisions": None,
                    "p95time": None,
                }
            )
            continue

        rows.append(
            {
                "Time_Total 预算": f"< {time_budget:.0f}",
                "满足条件数": match_count,
                "最高 RPS": round(best_row["RPS"], 2),
                "Index_Type": best_row.get("Index_Type"),
                "Time_Total": int(best_row["Time_Total"]),
                "Precisions": round(best_row["Precisions"], 5),
                "p95time": round(best_row["p95time"], 5),
            }
        )

    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(summary_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 Time_Total 预算和 Precisions 阈值筛选，找出 RPS 最高的配置"
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="输入 xlsx 文件路径",
    )
    parser.add_argument(
        "--time-budgets",
        nargs="+",
        type=float,
        default=DEFAULT_TIME_BUDGETS,
        help="Time_Total 预算列表，筛选 Time_Total < budget 的配置",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    print("=" * 70)
    print(f"数据集: {file_path.name}")
    print(f"Precisions 阈值: > {PRECISION_THRESHOLD:.2f}")
    print("=" * 70)

    df = load_dataset(file_path)
    print(f"总行数: {len(df)}")

    display_cols = [col for col in DISPLAY_COLUMNS if col in df.columns]

    for time_budget in args.time_budgets:
        print("\n" + "-" * 70)
        print(
            f"Time_Total < {time_budget:.0f} 且 Precisions > {PRECISION_THRESHOLD:.2f} 时，"
            "RPS 最高的配置"
        )
        print("-" * 70)

        match_count = count_matching_configs(df, PRECISION_THRESHOLD, time_budget)
        print(f"满足条件的配置数量: {match_count}")

        best_row = find_best_rps_by_time_budget(df, PRECISION_THRESHOLD, time_budget)
        if best_row is None:
            print(
                f"警告: 没有配置满足 Time_Total < {time_budget:.0f} 且 "
                f"Precisions > {PRECISION_THRESHOLD:.2f}"
            )
            continue

        print(best_row[display_cols].to_string())

    summary_df = build_summary_table(df, PRECISION_THRESHOLD, args.time_budgets)
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
