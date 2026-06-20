#!/bin/bash
#===============================================================================
# Missing Pretraining Strategies  (10-seed)
#
# eval_full.py 에는 포함되어 있으나 run_final_s2_10seed.sh 에서 제외된 5가지 전략:
#   v2_pt_ssp         — SSP only
#   v2_pt_cpcl        — CPCL only
#   v2_pt_bsj         — BSJ_MLM only
#   v2_pt_pairing     — Pairing only
#   v2_pt_mlm_cpcl_ssp — MLM + CPCL + SSP
#
# 각 전략: pretrain (seed=42) → finetune × 10 seeds
# Skip logic: pretrain은 model.pth 존재 여부, finetune은 training.json 존재 여부
#
# Usage:
#   ./scripts/final_v2/run_missing_pretraining_10seed.sh [GPU_ID]
#   GPU_ID 기본값: 0
#
# SEEDS_OVERRIDE 예시:
#   SEEDS_OVERRIDE="1 2 3 4 5" ./scripts/final_v2/run_missing_pretraining_10seed.sh 0
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

TASK="sites"
PT_SEED=42

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4

# Pretraining hyperparams
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

# Finetuning hyperparams
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Missing Pretraining Strategies  (GPU $GPU)"
echo "  Seeds: ${SEEDS[*]}"
echo "========================================"

run_pretrain() {
    local STRATEGY=$1; shift
    local PT_EXP="v2_ptm_${STRATEGY}"
    TOTAL=$((TOTAL+1))
    local PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[SKIP] pretrain: $PT_EXP"
        SKIPPED=$((SKIPPED+1))
        return 0
    fi
    RAN=$((RAN+1)); echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac \
        --data_file "$PT_DATA" \
        --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose "$@" \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
}

run_finetune() {
    local STRATEGY=$1
    local PT_PATH=$2
    for SEED in "${SEEDS[@]}"; do
        local EXP="v2_pt_${STRATEGY}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] finetune: $EXP"
            SKIPPED=$((SKIPPED+1))
            continue
        fi
        RAN=$((RAN+1)); echo "[RUN]  finetune: $EXP"
        FT_CMD="python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
            --device $GPU --exp $EXP \
            --interaction cross_attention --verbose"
        [ "$PT_PATH" != "none" ] && FT_CMD="$FT_CMD --load_pretrained $PT_PATH"
        eval "$FT_CMD" 2>&1 | tee "logs/v2/pt/${EXP}.log"
    done
}

echo ""
echo "━━━ [1/5] SSP ━━━"
run_pretrain "ssp" --ssp
PT_PATH="saved_models/circmac/v2_ptm_ssp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ssp" "$PT_PATH"

echo ""
echo "━━━ [2/5] CPCL ━━━"
run_pretrain "cpcl" --cpcl
PT_PATH="saved_models/circmac/v2_ptm_cpcl/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "cpcl" "$PT_PATH"

echo ""
echo "━━━ [3/5] BSJ ━━━"
run_pretrain "bsj" --bsj_mlm
PT_PATH="saved_models/circmac/v2_ptm_bsj/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "bsj" "$PT_PATH"

echo ""
echo "━━━ [4/5] Pairing ━━━"
run_pretrain "pairing" --pairing
PT_PATH="saved_models/circmac/v2_ptm_pairing/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "pairing" "$PT_PATH"

echo ""
echo "━━━ [5/5] MLM + CPCL + SSP ━━━"
run_pretrain "mlm_cpcl_ssp" --mlm --cpcl --ssp
PT_PATH="saved_models/circmac/v2_ptm_mlm_cpcl_ssp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_cpcl_ssp" "$PT_PATH"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
