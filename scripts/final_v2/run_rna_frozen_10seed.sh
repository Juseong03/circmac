#!/bin/bash
#===============================================================================
# EXP1 — RNA-LM Frozen Backbone  (10 seeds)
#
# Models: rnabert, rnaernie, rnafm, rnamsm
# Exp naming: exp1_fair_frozen_{model}_s{seed}   ← eval_full.py 기대값과 일치
#
# Per-model settings:
#   rnabert  (125M): max_len=438,  bs=128  ~  2h/seed → 20h  (seeds 1-10)
#   rnaernie (125M): max_len=511,  bs=128  ~  2h/seed → 20h
#   rnafm    (640M): max_len=1022, bs=64   ~  3h/seed → 30h
#   rnamsm   (100M): max_len=1022, bs=64   ~  2.5h/seed → 25h
#
# SEEDS_OVERRIDE 예시:
#   SEEDS_OVERRIDE="4 5 6 7 8 9 10" ./scripts/final_v2/run_rna_frozen_10seed.sh 0
#
# Usage: ./scripts/final_v2/run_rna_frozen_10seed.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

declare -A ML_MAP=( [rnabert]=438  [rnaernie]=511  [rnafm]=1022 [rnamsm]=1022 )
declare -A BS_MAP=( [rnabert]=128  [rnaernie]=128  [rnafm]=64   [rnamsm]=64   )

mkdir -p logs/v2/rna_frozen saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  RNA-LM Frozen  (GPU $GPU)"
echo "  Seeds: ${SEEDS[*]}"
echo "========================================"

for MODEL in rnabert rnaernie rnafm rnamsm; do
    ML=${ML_MAP[$MODEL]}
    BS=${BS_MAP[$MODEL]}
    echo ""
    echo "━━━ $MODEL  (max_len=$ML, bs=$BS) ━━━"
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_fair_frozen_${MODEL}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py \
            --model_name "$MODEL" --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $ML \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/v2/rna_frozen/${EXP}.log"
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
