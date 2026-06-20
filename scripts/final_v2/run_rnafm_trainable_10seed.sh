#!/bin/bash
#===============================================================================
# RNA-FM Trainable (Fine-tuned) Backbone  (10 seeds)
#
# eval_full.py 에 exp1_fair_trainable_rnafm 로 등록되어 있으나
# run_rna_trainable_10seed.sh 에서 제외된 모델 (메모리 부담으로 별도 스크립트화).
#
# RNA-FM (640M): max_len=1022, bs=8  ~  8h/seed → 80h (seeds 1-10)
#
# Exp naming: exp1_fair_trainable_rnafm_s{seed}
#
# Usage:
#   ./scripts/final_v2/run_rnafm_trainable_10seed.sh [GPU_ID]
#   GPU_ID 기본값: 0
#
# SEEDS_OVERRIDE 예시:
#   SEEDS_OVERRIDE="1 2 3 4 5" ./scripts/final_v2/run_rnafm_trainable_10seed.sh 0
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

MODEL="rnafm"
TASK="sites"
MAX_LEN=1022; BS=8
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

mkdir -p logs/v2/rna_trainable saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  RNA-FM Trainable  (GPU $GPU)"
echo "  max_len=$MAX_LEN  bs=$BS"
echo "  Seeds: ${SEEDS[*]}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    EXP="exp1_fair_trainable_rnafm_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --trainable_pretrained \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/v2/rna_trainable/${EXP}.log"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
