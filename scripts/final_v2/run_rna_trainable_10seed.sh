#!/bin/bash
#===============================================================================
# EXP1 — RNA-LM Trainable (Fine-tuned) Backbone  (10 seeds)
#
# Models: rnabert, rnaernie, rnamsm   (rnafm trainable 제외 — 논문 Figure 미포함)
# Exp naming: exp1_fair_trainable_{model}_s{seed}   ← eval_full.py 기대값과 일치
#
# Per-model settings:
#   rnabert  (125M): max_len=438,  bs=32  ~  4h/seed → 40h  (seeds 1-10)
#   rnaernie (125M): max_len=511,  bs=32  ~  4h/seed → 40h
#   rnamsm   (100M): max_len=1022, bs=8   ~  8h/seed → 80h  ← bottleneck
#
# 권장: rnamsm은 별도 GPU에서 병렬 실행
#   MODELS_OVERRIDE="rnabert rnaernie" ./run_rna_trainable_10seed.sh 0
#   MODELS_OVERRIDE="rnamsm"           ./run_rna_trainable_10seed.sh 1
#
# SEEDS_OVERRIDE 예시:
#   SEEDS_OVERRIDE="4 5 6 7 8 9 10" ./scripts/final_v2/run_rna_trainable_10seed.sh 0
#
# Usage: ./scripts/final_v2/run_rna_trainable_10seed.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi
if [ -n "${MODELS_OVERRIDE:-}" ]; then
    read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
    MODELS=(rnabert rnaernie rnamsm)
fi

TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

declare -A ML_MAP=( [rnabert]=438  [rnaernie]=511  [rnamsm]=1022 )
declare -A BS_MAP=( [rnabert]=32   [rnaernie]=32   [rnamsm]=8    )

mkdir -p logs/v2/rna_trainable saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  RNA-LM Trainable  (GPU $GPU)"
echo "  Models: ${MODELS[*]}"
echo "  Seeds:  ${SEEDS[*]}"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    ML=${ML_MAP[$MODEL]}
    BS=${BS_MAP[$MODEL]}
    echo ""
    echo "━━━ $MODEL  (max_len=$ML, bs=$BS) ━━━"
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_fair_trainable_${MODEL}_s${SEED}"
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
            --trainable_pretrained \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/v2/rna_trainable/${EXP}.log"
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
