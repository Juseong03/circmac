#!/bin/bash
#===============================================================================
# GPU 3 — EXP1 RNA-LM: Frozen + Trainable
#
# 실험:
#   RNA Frozen  : rnabert(438), rnaernie(511), rnafm(1022), rnamsm(1022)
#                 exp naming: exp1_fair_frozen_{model}_s{seed}
#   RNA Trainable: rnabert(438), rnaernie(511), rnamsm(1022)  [rnafm 제외]
#                 exp naming: exp1_fair_trainable_{model}_s{seed}
#
# Seeds: 1 2 3 4 5 6 7 8 9 10  (기존 seed는 skip 처리)
#
# 예상 소요:
#   Frozen    : ~95h  (새 seed 기준 ~66h)
#   Trainable : ~160h (새 seed 기준 ~112h)  ← rnamsm이 bottleneck (~8h/seed)
#   합계      : ~250h
#
# Usage:
#   ./scripts/final_v2/run_gpu3_10seeds.sh [GPU_ID]
#   GPU_ID 기본값: 3
#===============================================================================
set -e

GPU=${1:-3}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  GPU $GPU | RNA-LM Frozen + Trainable"
echo "  Seeds: 1-10"
echo "========================================"

# ── Phase 1: RNA-LM Frozen ────────────────────────────────────────────────────
echo ""
echo "  [Phase 1] RNA Frozen (rnabert, rnaernie, rnafm, rnamsm)..."
SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash scripts/final_v2/run_rna_frozen_10seed.sh "$GPU"

# ── Phase 2: RNA-LM Trainable ─────────────────────────────────────────────────
echo ""
echo "  [Phase 2] RNA Trainable (rnabert, rnaernie, rnamsm)..."
SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash scripts/final_v2/run_rna_trainable_10seed.sh "$GPU"

echo ""
echo "  GPU $GPU 완료!"
