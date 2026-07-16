#!/usr/bin/env python3
"""Compute ranking loss for SCANN sweep results grouped by sample_ratio and reorder_k."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_ranking_loss(x, y) -> float:
    """
    计算两个序列之间的 Ranking Loss (逆序率)
    x: 在大数据集上的性能指标序列 (如 original_rps)
    y: 在小数据集上的性能指标序列 (如 sampled_rps)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    discordant = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] > x[j] and y[i] < y[j]) or (x[i] < x[j] and y[i] > y[j]):
                discordant += 1
            total_pairs += 1

    return discordant / total_pairs if total_pairs > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read sampling_scann_param_sweep_results.xlsx and compute ranking loss "
            "for RPS and precision within each (sample_ratio, reorder_k) group."
        )
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-xlsx",
        default=str(script_dir / "sampling_scann_param_sweep_results.xlsx"),
        help="Input sweep results xlsx path.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(script_dir / "scann_ratio_reorder_k_ranking_loss.csv"),
        help="Output csv path for grouped ranking loss.",
    )
    return parser.parse_args()


def load_valid_rows(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    required_cols = [
        "sample_ratio",
        "reorder_k",
        "original_rps",
        "sampled_rps",
        "original_mean_precisions",
        "sampled_mean_precisions",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {xlsx_path}: {missing}")

    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    metric_cols = [
        "original_rps",
        "sampled_rps",
        "original_mean_precisions",
        "sampled_mean_precisions",
    ]
    df = df.dropna(subset=["sample_ratio", "reorder_k", *metric_cols]).copy()
    return df


def main() -> None:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx).resolve()
    output_csv = Path(args.output_csv).resolve()

    df = load_valid_rows(input_xlsx)

    print("================ 全局 Ranking Loss ================")
    overall_rps_loss = calculate_ranking_loss(df["original_rps"], df["sampled_rps"])
    overall_prec_loss = calculate_ranking_loss(
        df["original_mean_precisions"], df["sampled_mean_precisions"]
    )
    print(f"全局吞吐量 (RPS) Ranking Loss: {overall_rps_loss:.4%}")
    print(f"全局准确率 (Precision) Ranking Loss: {overall_prec_loss:.4%}\n")

    print("======= 按 sample_ratio + reorder_k 拆分的 Ranking Loss =======")
    rows = []
    grouped = df.groupby(["sample_ratio", "reorder_k"], sort=True)
    for (sample_ratio, reorder_k), group in grouped:
        orig_rps = group["original_rps"].values
        samp_rps = group["sampled_rps"].values
        orig_prec = group["original_mean_precisions"].values
        samp_prec = group["sampled_mean_precisions"].values

        rps_loss = calculate_ranking_loss(orig_rps, samp_rps)
        prec_loss = calculate_ranking_loss(orig_prec, samp_prec)

        print(
            f"sample_ratio={sample_ratio}, reorder_k={reorder_k} "
            f"(样本数: {len(group)} 行):"
        )
        print(f"  - 吞吐量 (RPS) Ranking Loss: {rps_loss:.2%}")
        print(f"  - 准确率 (Precision) Ranking Loss: {prec_loss:.2%}")
        print("-" * 50)

        rows.append(
            {
                "sample_ratio": sample_ratio,
                "reorder_k": reorder_k,
                "num_rows": len(group),
                "rps_ranking_loss": rps_loss,
                "precision_ranking_loss": prec_loss,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved grouped ranking loss csv: {output_csv}")


if __name__ == "__main__":
    main()
