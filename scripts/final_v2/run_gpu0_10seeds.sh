#!/bin/bash
#===============================================================================
# GPU 0 — EXP1 Base Encoder + EXP5 Interaction + EXP6 Site Head
#
# 실험:
#   EXP1 Base : circmac, mamba, hymba, lstm, transformer  (v2_enc_*)
#   EXP5      : cross_attn, concat, elementwise           (v2_int_*)
#   EXP6      : conv1d, linear                            (v2_head_*)
#
# Seeds: 1 2 3 4 5 6 7 8 9 10  (기존 seed는 skip 처리)
#
# Usage:
#   ./scripts/final_v2/run_gpu0_10seeds.sh [GPU_ID]
#   GPU_ID 기본값: 0
#===============================================================================
set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  GPU $GPU | EXP1 Base + EXP5 + EXP6"
echo "  Seeds: 1-10"
echo "========================================"

SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash scripts/final_v2/run_final_s1.sh "$GPU"

echo ""
echo "  GPU $GPU 완료!"
