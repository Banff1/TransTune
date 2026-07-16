#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OtterTune / VDTuner 调优 JSON → xlsx。

默认：与 auto-configure/vdtuner/prior/*.xlsx 相同的宽表布局（22 列），
输入默认 ottertune-configure/log/random-match-int-100-angular-no-filters.json。

可选 --legacy-normalize：旧版 json_normalize 带点号列名的展平方式。
"""
import argparse
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("需要安装 pandas 与 openpyxl: pip install pandas openpyxl", file=sys.stderr)
    raise

# 与 random-100-match-kw-small-vocab-no-filters.xlsx 一致的列名与顺序
PRIOR_COLUMNS = [
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
    "Time_Step",
    "RPS",
]

# 旧版：顶层标量优先
_META_ORDER = ["iteration", "time", "precisions", "p95time", "Time", "RPS"]


def _bool_to_01(v):
    if v is True or v == 1 or v == "1":
        return 1
    if v is False or v == 0 or v == "0":
        return 0
    return v


def tuning_record_to_prior_row(rec):
    """一条 JSON 对象 → 与 vdtuner/prior/*.xlsx 对齐的一行 dict。"""
    ic = rec.get("index_conf") or {}
    sc = rec.get("system_conf") or {}
    return {
        "Iteration": rec.get("iteration"),
        "Time_Total": rec.get("time"),
        "Index_Type": ic.get("index_type"),
        "nlist": ic.get("nlist"),
        "nprobe": ic.get("nprobe"),
        "m": ic.get("m"),
        "nbits": ic.get("nbits"),
        "M": ic.get("M"),
        "efConstruction": ic.get("efConstruction"),
        "ef": ic.get("ef"),
        "reorder_k": ic.get("reorder_k"),
        "maxSize": sc.get("dataCoord*segment*maxSize"),
        "sealProportion": sc.get("dataCoord*segment*sealProportion"),
        "autoHandoff": _bool_to_01(sc.get("queryCoord*autoHandoff")),
        "autoBalance": _bool_to_01(sc.get("queryCoord*autoBalance")),
        "gracefulTime": sc.get("common*gracefulTime"),
        "insertBufSize": sc.get("dataNode*segment*insertBufSize"),
        "minSegmentSizeToIndex": sc.get("rootCoord*minSegmentSizeToEnableIndex"),
        "Precisions": rec.get("precisions"),
        "p95time": rec.get("p95time"),
        "Time_Step": rec.get("Time"),
        "RPS": rec.get("RPS"),
    }


def json_tuning_to_prior_dataframe(data):
    if not isinstance(data, list):
        raise ValueError("JSON root must be an array of objects")
    rows = [tuning_record_to_prior_row(r) for r in data if isinstance(r, dict)]
    if not rows:
        return pd.DataFrame(columns=PRIOR_COLUMNS)
    df = pd.DataFrame(rows)
    for c in PRIOR_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[PRIOR_COLUMNS]


def infer_meta_columns(data):
    seen = set()
    for row in data:
        if isinstance(row, dict):
            for k in row.keys():
                seen.add(k)
    return [k for k in _META_ORDER if k in seen] + sorted(
        k for k in seen if k not in _META_ORDER and k not in ("index_conf", "system_conf")
    )


def json_tuning_to_dataframe_legacy_normalize(data):
    if not isinstance(data, list):
        raise ValueError("JSON root must be an array of objects")
    if len(data) == 0:
        return pd.DataFrame()
    sep = "."
    df = pd.json_normalize(data, sep=sep)
    meta = infer_meta_columns(data)
    present_meta = [c for c in meta if c in df.columns]
    rest = sorted(c for c in df.columns if c not in present_meta)
    return df[present_meta + rest]


def convert_json_to_excel(json_file_path, excel_file_path, legacy_normalize=False):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if legacy_normalize:
        df = json_tuning_to_dataframe_legacy_normalize(data)
    else:
        df = json_tuning_to_prior_dataframe(data)
    out_dir = os.path.dirname(os.path.abspath(excel_file_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_excel(excel_file_path, index=False, engine="openpyxl")
    print("转换成功: {}".format(excel_file_path))
    print("行数: {}".format(len(df)))
    print("列数: {}".format(len(df.columns)))


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_json = os.path.abspath(
        os.path.join(_script_dir, "..", "log", "random-match-int-100-angular-no-filters.json")
    )
    parser = argparse.ArgumentParser(
        description="调优 JSON → xlsx（默认列布局同 vdtuner/prior/*.xlsx）"
    )
    parser.add_argument(
        "--json",
        "-j",
        default=_default_json,
        help="输入 JSON（默认 log/random-match-int-100-angular-no-filters.json）",
    )
    parser.add_argument(
        "--xlsx",
        "-o",
        default=None,
        help="输出 xlsx（默认与 JSON 同路径、扩展名 .xlsx）",
    )
    parser.add_argument(
        "--legacy-normalize",
        action="store_true",
        help="使用旧版 index_conf.xxx / system_conf.xxx 点号展平列名",
    )
    args = parser.parse_args()
    json_path = os.path.abspath(args.json)
    if args.xlsx:
        xlsx_path = os.path.abspath(args.xlsx)
    else:
        base, _ = os.path.splitext(json_path)
        xlsx_path = base + ".xlsx"
    if not os.path.isfile(json_path):
        print("找不到 JSON 文件: {}".format(json_path), file=sys.stderr)
        sys.exit(1)
    try:
        convert_json_to_excel(json_path, xlsx_path, legacy_normalize=args.legacy_normalize)
    except Exception as e:
        print("转换失败: {}".format(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
