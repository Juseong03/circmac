#!/bin/bash
#===============================================================================
# GPU 1 — EXP4 CircMAC Ablation Study
#
# 실험:
#   EXP4: full, no_attn, no_mamba, no_conv, no_circ_bias,
#         attn_only, mamba_only, cnn_only               (v2_abl_*)
#
# Seeds: 1 2 3 4 5 6 7 8 9 10  (기존 seed는 skip 처리)
#
# Usage:
#   ./scripts/final_v2/run_gpu1_10seeds.sh [GPU_ID]
#   GPU_ID 기본값: 1
#===============================================================================
set -e

GPU=${1:-1}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  GPU $GPU | EXP4 Ablation"
echo "  Seeds: 1-10"
echo "========================================"

SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash scripts/final_v2/run_final_s3.sh "$GPU"

echo ""
echo "  GPU $GPU 완료!"
