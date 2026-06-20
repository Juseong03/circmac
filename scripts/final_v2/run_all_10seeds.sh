#!/bin/bash
#===============================================================================
# run_all_10seeds.sh  —  전체 실험 10-seed 확장 launcher
#
# 대상 실험:
#   EXP1 Base   : circmac, mamba, hymba, lstm, transformer  (v2_enc_*)
#   EXP1 RNA Frz: rnabert, rnaernie, rnafm, rnamsm frozen   (exp1_fair_frozen_*)
#   EXP1 RNA FT : rnabert, rnaernie, rnamsm trainable       (exp1_fair_trainable_*)
#   EXP2 PT Str : 7 pretraining strategies                  (v2_pt_*)
#   EXP4 Ablat  : 8 CircMAC ablation variants               (v2_abl_*)
#   EXP5 Intxn  : cross_attn, concat, elementwise           (v2_int_*)
#   EXP6 Head   : conv1d, linear                            (v2_head_*)
#
# GPU 할당 (8 GPU 기준 권장):
#   GPU 0 : EXP1 Base + EXP5 + EXP6           (run_final_s1.sh)
#   GPU 1 : EXP2 pretrain-only                (run_final_s2_10seed.sh, PRETRAIN_ONLY=1)
#   GPU 2 : EXP2 finetune seeds 1-5           (run_final_s2_10seed.sh, SKIP_PRETRAIN=1)
#   GPU 3 : EXP2 finetune seeds 6-10          (run_final_s2_10seed.sh, SKIP_PRETRAIN=1)
#   GPU 4 : EXP4 ablation seeds 1-5           (run_final_s3.sh)
#   GPU 5 : EXP4 ablation seeds 6-10          (run_final_s3.sh)
#   GPU 6 : RNA frozen (rnabert, rnaernie, rnafm, rnamsm)
#   GPU 7 : RNA trainable (rnabert, rnaernie)
#   GPU 8 : RNA trainable (rnamsm)  ← 별도 GPU 권장
#
# 보유 GPU가 적을 경우:
#   - RNA trainable (rnamsm)을 GPU 7과 합치거나 순차 실행
#   - EXP4 ablation을 단일 GPU에서 순차 실행
#
# Usage:
#   ./scripts/final_v2/run_all_10seeds.sh [NUM_GPUS]
#   NUM_GPUS: 사용할 GPU 수 (기본: 8)
#===============================================================================
set -e

NUM_GPUS=${1:-8}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  All Experiments — 10-seed Launcher"
echo "  NUM_GPUS=$NUM_GPUS"
echo "========================================"
echo ""

SCRIPTS="scripts/final_v2"

# ── GPU 0: EXP1 Base + EXP5 Interaction + EXP6 Head ─────────────────────────
echo "[GPU 0] EXP1 Base + EXP5 + EXP6  (seeds 1-10)"
SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
    bash $SCRIPTS/run_final_s1.sh 0 &
PID0=$!

# ── GPU 1-3: EXP2 Pretraining Strategy ───────────────────────────────────────
echo "[GPU 1] EXP2 pretrain-only"
PRETRAIN_ONLY=1 \
    bash $SCRIPTS/run_final_s2_10seed.sh 1
# pretrain 완료 후 finetune 병렬 시작

if [ "$NUM_GPUS" -ge 4 ]; then
    echo "[GPU 2] EXP2 finetune seeds 1-5"
    SKIP_PRETRAIN=1 SEEDS_OVERRIDE="1 2 3 4 5" \
        bash $SCRIPTS/run_final_s2_10seed.sh 2 &
    PID2=$!
    echo "[GPU 3] EXP2 finetune seeds 6-10"
    SKIP_PRETRAIN=1 SEEDS_OVERRIDE="6 7 8 9 10" \
        bash $SCRIPTS/run_final_s2_10seed.sh 3 &
    PID3=$!
else
    echo "[GPU 1] EXP2 finetune all seeds (sequential)"
    SKIP_PRETRAIN=1 SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_final_s2_10seed.sh 1 &
    PID2=$!; PID3=$PID2
fi

# ── GPU 4-5: EXP4 Ablation ───────────────────────────────────────────────────
if [ "$NUM_GPUS" -ge 6 ]; then
    echo "[GPU 4] EXP4 ablation seeds 1-5"
    SEEDS_OVERRIDE="1 2 3 4 5" \
        bash $SCRIPTS/run_final_s3.sh 4 &
    PID4=$!
    echo "[GPU 5] EXP4 ablation seeds 6-10"
    SEEDS_OVERRIDE="6 7 8 9 10" \
        bash $SCRIPTS/run_final_s3.sh 5 &
    PID5=$!
else
    echo "[GPU 0] EXP4 ablation all seeds (sequential, after EXP1/5/6)"
    wait $PID0
    SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_final_s3.sh 0 &
    PID4=$!; PID5=$PID4
fi

# ── GPU 6: RNA-LM Frozen ─────────────────────────────────────────────────────
if [ "$NUM_GPUS" -ge 7 ]; then
    echo "[GPU 6] RNA-LM Frozen (all models, seeds 1-10)"
    SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_rna_frozen_10seed.sh 6 &
    PID6=$!
else
    echo "[GPU 0] RNA-LM Frozen (sequential)"
    wait $PID0 2>/dev/null || true
    SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_rna_frozen_10seed.sh 0 &
    PID6=$!
fi

# ── GPU 7-8: RNA-LM Trainable ────────────────────────────────────────────────
if [ "$NUM_GPUS" -ge 9 ]; then
    echo "[GPU 7] RNA Trainable: rnabert, rnaernie (seeds 1-10)"
    MODELS_OVERRIDE="rnabert rnaernie" SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_rna_trainable_10seed.sh 7 &
    PID7=$!
    echo "[GPU 8] RNA Trainable: rnamsm (seeds 1-10)"
    MODELS_OVERRIDE="rnamsm" SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" \
        bash $SCRIPTS/run_rna_trainable_10seed.sh 8 &
    PID8=$!
elif [ "$NUM_GPUS" -ge 8 ]; then
    echo "[GPU 7] RNA Trainable: all models, seeds 1-5"
    SEEDS_OVERRIDE="1 2 3 4 5" \
        bash $SCRIPTS/run_rna_trainable_10seed.sh 7 &
    PID7=$!
    echo "[GPU 7] RNA Trainable: seeds 6-10 (after seeds 1-5)"
    wait $PID7 2>/dev/null || true
    SEEDS_OVERRIDE="6 7 8 9 10" \
        bash $SCRIPTS/run_rna_trainable_10seed.sh 7 &
    PID8=$!
else
    echo "[NOTE] RNA Trainable은 별도로 실행하세요:"
    echo "  SEEDS_OVERRIDE=\"4 5 6 7 8 9 10\" bash $SCRIPTS/run_rna_trainable_10seed.sh [GPU]"
    PID7=0; PID8=0
fi

echo ""
echo "========================================"
echo "  All workers launched. Waiting..."
echo "========================================"

wait $PID0 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 2>/dev/null || true

echo ""
echo "========================================"
echo "  All experiments finished!"
echo "  → eval 실행: python scripts/eval_full.py --group all"
echo "========================================"
