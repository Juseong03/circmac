#!/bin/bash
#===============================================================================
# GPU 2 — EXP2 Pretraining Strategy Comparison
#
# 실험:
#   EXP2: nopt, mlm, ntp, mlm_ssp, mlm_cpcl, mlm_ntp, all  (v2_pt_*)
#
#   Phase 1 (pretrain): 6 strategies × seed=42 — 한 번만 실행
#   Phase 2 (finetune): 7 strategies × 10 seeds = 70 runs
#
# Seeds: 1 2 3 4 5 6 7 8 9 10  (기존 seed는 skip 처리)
#
# Usage:
#   ./scripts/final_v2/run_gpu2_10seeds.sh [GPU_ID]
#   GPU_ID 기본값: 2
#===============================================================================
set -e

GPU=${1:-2}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  GPU $GPU | EXP2 Pretraining Strategy"
echo "  Seeds: 1-10"
echo "========================================"

# Phase 1: pretrain-only (이미 완료된 경우 자동 skip)
echo ""
echo "  [Phase 1] Pretraining..."
PRETRAIN_ONLY=1 \
    bash scripts/final_v2/run_final_s2_10seed.sh "$GPU"

# Phase 2: finetune (pretrain 완료 후 실행)
echo ""
echo "  [Phase 2] Finetuning seeds 1-10..."
SKIP_PRETRAIN=1 SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash scripts/final_v2/run_final_s2_10seed.sh "$GPU"

echo ""
echo "  GPU $GPU 완료!"
