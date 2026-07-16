#!/usr/bin/env bash
set -euo pipefail

BASE_SEED=42
PER_ROUND=2
TARGETS=(20 50 100)

QPP_DATASET="random-match-int-2048-angular-no-filters"
PCR_DATASET="random-100-match-kw-small-vocab-no-filters"
PRIOR_CONFIG="example_vdtuner_prior.json"

PREV=0

for CUR in "${TARGETS[@]}"; do
  DELTA=$((CUR - PREV))
  if [ "$DELTA" -le 0 ]; then
    echo "skip target=$CUR (PREV=$PREV)"
    continue
  fi

  # 增量seed偏移：BASE_SEED + 已完成轮数
  SEED=$((BASE_SEED + PREV))
  LOG="online_r${CUR}.log"

  echo "=== run target_rounds=$CUR, delta=$DELTA, seed=$SEED ==="

  nohup python run_vdb_two_stage_full_pipeline.py online \
    --prior-config "$PRIOR_CONFIG" \
    --qpp-dataset "$QPP_DATASET" \
    --pcr-dataset "$PCR_DATASET" \
    --overwrite-train-data \
    --active-rounds "$DELTA" \
    --active-per-round "$PER_ROUND" \
    --lhs-seed "$SEED" \
    --dsd-threshold 1.0 \
    --force-active-collect \
    --engine milvus-single-node \
    --process-tag milvus-p10 \
    --dipredict-layer-sizes-auto \
    > "$LOG" 2>&1

  # 归档本轮推荐结果（避免下轮被覆盖）
  RES_DIR="../parameter_configuration_recommend/[128, 256, 256, 64]_[256, 256, 256, 64]_TD3_prior/main/recommend_results"
  LATEST=$(ls -t "$RES_DIR"/eval_*_vdb_${PCR_DATASET}_*.csv | head -n 1 || true)
  if [ -n "${LATEST:-}" ]; then
    cp "$LATEST" "$RES_DIR/eval_${PCR_DATASET}_rounds_${CUR}.csv"
  fi

  PREV=$CUR
done