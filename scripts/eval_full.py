#!/usr/bin/env python3
"""
eval_full.py
saved_models/ 의 모든 실험을 train / val / test 세 split에 대해 재평가.

Output:
  eval_results/eval_full_summary.csv          — 모든 metrics (split별, seed별)
  eval_results/preds/{exp_tpl}_s{seed}/
      train_preds.pkl  /  val_preds.pkl  /  test_preds.pkl
      (DataFrame: sample_idx, position, label, prob)

Usage:
    python scripts/eval_full.py --device 0
    python scripts/eval_full.py --device 0 --group encoder
    python scripts/eval_full.py --device 0 --group pretraining --skip_preds
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from utils import prepare_datasets
from utils_config import get_model_config

# ── 경로 ──────────────────────────────────────────────────────────────────────
SAVED    = ROOT / "saved_models"
OUT      = ROOT / "eval_results"
PRED_DIR = OUT / "preds"
OUT.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
MAX_LEN = 1022
D_MODEL = 128
N_LAYER = 6
BS      = 32
LM_BS   = 8    # rnamsm, rnafm: OOM with BS=32
WORKERS = 4
SEEDS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

LM_MAX_LEN = {
    "rnabert":  438,
    "rnaernie": 511,
    "rnamsm":   1022,
    "rnafm":    1022,
    "circmac":  1022,
}

# ── 실험 목록 ──────────────────────────────────────────────────────────────────
# (group, label, model_name, exp_template, interaction, trainable_pt)
EXPERIMENTS = [
    # Encoder comparison
    ("encoder", "LSTM",        "lstm",        "v2_enc_lstm",        "cross_attention", False),
    ("encoder", "Transformer", "transformer", "v2_enc_transformer", "cross_attention", False),
    ("encoder", "Mamba",       "mamba",       "v2_enc_mamba",       "cross_attention", False),
    ("encoder", "Hymba",       "hymba",       "v2_enc_hymba",       "cross_attention", False),
    ("encoder", "CircMAC",     "circmac",     "v2_abl_full",        "cross_attention", False),

    # Pretrained comparison
    ("pretrained", "RNABERT (frozen)",      "rnabert",  "exp1_fair_frozen_rnabert",     "cross_attention", False),
    ("pretrained", "RNABERT (fine-tuned)",  "rnabert",  "exp1_fair_trainable_rnabert",  "cross_attention", True),
    ("pretrained", "RNAErnie (frozen)",     "rnaernie", "exp1_fair_frozen_rnaernie",    "cross_attention", False),
    ("pretrained", "RNAErnie (fine-tuned)", "rnaernie", "exp1_fair_trainable_rnaernie", "cross_attention", True),
    ("pretrained", "RNAMSM (frozen)",       "rnamsm",   "exp1_fair_frozen_rnamsm",      "cross_attention", False),
    ("pretrained", "RNAMSM (fine-tuned)",   "rnamsm",   "exp1_fair_trainable_rnamsm",   "cross_attention", True),
    ("pretrained", "RNA-FM (frozen)",       "rnafm",    "exp1_fair_frozen_rnafm",       "cross_attention", False),
    ("pretrained", "RNA-FM (fine-tuned)",   "rnafm",    "exp1_fair_trainable_rnafm",    "cross_attention", True),
    ("pretrained", "CircMAC (NoPT)",        "circmac",  "v2_abl_full",                  "cross_attention", False),
    ("pretrained", "CircMAC (Pairing)",     "circmac",  "v2_pt_pairing",                "cross_attention", False),

    # Ablation — modules
    ("ablation", "CircMAC (full)",   "circmac", "v2_abl_full",         "cross_attention", False),
    ("ablation", "Attn only",        "circmac", "v2_abl_attn_only",    "cross_attention", False),
    ("ablation", "Mamba only",       "circmac", "v2_abl_mamba_only",   "cross_attention", False),
    ("ablation", "CNN only",         "circmac", "v2_abl_cnn_only",     "cross_attention", False),
    ("ablation", "No Attn",          "circmac", "v2_abl_no_attn",      "cross_attention", False),
    ("ablation", "No Mamba",         "circmac", "v2_abl_no_mamba",     "cross_attention", False),
    ("ablation", "No Conv",          "circmac", "v2_abl_no_conv",      "cross_attention", False),
    ("ablation", "No CircBias",      "circmac", "v2_abl_no_circ_bias", "cross_attention", False),

    # Interaction mechanism
    ("interaction", "Concat",      "circmac", "v2_int_concat",      "concat",          False),
    ("interaction", "Elementwise", "circmac", "v2_int_elementwise", "elementwise",     False),
    ("interaction", "Cross-Attn",  "circmac", "v2_int_cross_attn",  "cross_attention", False),

    # Site head
    ("site_head", "Conv1D head", "circmac", "v2_head_conv1d", "cross_attention", False),
    ("site_head", "Linear head", "circmac", "v2_head_linear", "cross_attention", False),

    # Pretraining strategy
    ("pretraining", "NoPT",         "circmac", "v2_pt_nopt",         "cross_attention", False),
    ("pretraining", "MLM",          "circmac", "v2_pt_mlm",          "cross_attention", False),
    ("pretraining", "NTP",          "circmac", "v2_pt_ntp",          "cross_attention", False),
    ("pretraining", "SSP",          "circmac", "v2_pt_ssp",          "cross_attention", False),
    ("pretraining", "CPCL",         "circmac", "v2_pt_cpcl",         "cross_attention", False),
    ("pretraining", "BSJ",          "circmac", "v2_pt_bsj",          "cross_attention", False),
    ("pretraining", "MLM+NTP",      "circmac", "v2_pt_mlm_ntp",      "cross_attention", False),
    ("pretraining", "MLM+SSP",      "circmac", "v2_pt_mlm_ssp",      "cross_attention", False),
    ("pretraining", "MLM+CPCL",     "circmac", "v2_pt_mlm_cpcl",     "cross_attention", False),
    ("pretraining", "Pairing",      "circmac", "v2_pt_pairing",      "cross_attention", False),
    ("pretraining", "MLM+CPCL+SSP", "circmac", "v2_pt_mlm_cpcl_ssp","cross_attention", False),
    ("pretraining", "All",          "circmac", "v2_pt_all",          "cross_attention", False),
]


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────
def load_raw_data():
    df_train_raw = pd.read_pickle(ROOT / "data/df_train_final.pkl")
    df_test_raw  = pd.read_pickle(ROOT / "data/df_test_final.pkl")

    df_train_raw["length"] = df_train_raw["circRNA"].apply(len)
    df_test_raw["length"]  = df_test_raw["circRNA"].apply(len)

    df_train_raw = df_train_raw[df_train_raw["binding"] == 1].reset_index(drop=True)
    df_test_raw  = df_test_raw[df_test_raw["binding"]  == 1].reset_index(drop=True)

    return df_train_raw, df_test_raw


def build_datasets(df_train_raw, df_test_raw, seed, max_len):
    """seed 기반 train/val split + test dataset 생성"""
    ml = max_len
    df_tr = df_train_raw[df_train_raw["length"] <= ml].reset_index(drop=True)
    df_te = df_test_raw[df_test_raw["length"]   <= ml].reset_index(drop=True)

    train_ds, val_ds, test_ds, _ = prepare_datasets(
        df=df_tr, df_test=df_te,
        max_len=ml + 2, target="mirna", seed=seed, kmer=1,
    )
    return train_ds, val_ds, test_ds


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(labels, probs):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        accuracy_score, matthews_corrcoef,
    )

    n_total = len(labels)
    n_pos   = int(labels.sum())
    n_neg   = n_total - n_pos
    pos_rate = round(n_pos / n_total, 6) if n_total > 0 else float("nan")

    try:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    except Exception:
        auroc = auprc = float("nan")

    # threshold sweep (best F1_macro)
    best_t, best_f1mac = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        pb  = (probs >= t).astype(int)
        fm  = float(f1_score(labels, pb, average="macro", zero_division=0))
        if fm > best_f1mac:
            best_f1mac, best_t = fm, t

    pb = (probs >= best_t).astype(int)
    return {
        "n_tokens":   n_total,
        "n_pos":      n_pos,
        "n_neg":      n_neg,
        "pos_rate":   pos_rate,
        "auroc":      round(auroc, 6),
        "auprc":      round(auprc, 6),
        "threshold":  round(float(best_t), 4),
        "acc":        round(float(accuracy_score(labels, pb)), 6),
        "f1_macro":   round(float(f1_score(labels, pb, average="macro",   zero_division=0)), 6),
        "prec_macro": round(float(precision_score(labels, pb, average="macro", zero_division=0)), 6),
        "rec_macro":  round(float(recall_score(labels, pb, average="macro",    zero_division=0)), 6),
        "f1_pos":     round(float(f1_score(labels, pb, pos_label=1, zero_division=0)), 6),
        "prec_pos":   round(float(precision_score(labels, pb, pos_label=1, zero_division=0)), 6),
        "rec_pos":    round(float(recall_score(labels, pb, pos_label=1, zero_division=0)), 6),
        "mcc":        round(float(matthews_corrcoef(labels, pb)), 6),
    }


# ── Inference ─────────────────────────────────────────────────────────────────
def extract_preds(tensors):
    """tensors dict → (labels_flat, probs_flat, sample_idx_flat, pos_flat)"""
    preds_raw  = tensors["preds_sites"]   # (N, L-1, 2)
    labels_raw = tensors["labels_sites"]  # (N, L)

    labels_aligned = labels_raw[:, 1:]   # (N, L-1) — CLS 제거
    N, Lm1 = labels_aligned.shape

    # 예측 확률
    if preds_raw.ndim == 3 and preds_raw.shape[-1] == 2:
        probs_2d = torch.softmax(preds_raw.float(), dim=-1)[:, :, 1]  # (N, L-1)
    else:
        probs_2d = preds_raw.float().squeeze(-1)

    labels_flat  = labels_aligned.reshape(-1).numpy()
    probs_flat   = probs_2d.reshape(-1).numpy()
    sample_flat  = np.repeat(np.arange(N), Lm1)
    pos_flat     = np.tile(np.arange(Lm1), N)

    valid = labels_flat != -100
    return (labels_flat[valid].astype(np.int8),
            probs_flat[valid].astype(np.float32),
            sample_flat[valid],
            pos_flat[valid])


def run_split(trainer, loader, split_name):
    """단일 split inference → (metrics dict, preds DataFrame)"""
    _, tensors, _ = trainer.step_loader(loader, 0, is_train=False, data_type=split_name)

    if not isinstance(tensors.get("preds_sites"), torch.Tensor):
        return None, None

    labels, probs, sample_idx, positions = extract_preds(tensors)
    metrics = compute_metrics(labels, probs)

    preds_df = pd.DataFrame({
        "sample_idx": sample_idx,
        "position":   positions,
        "label":      labels,
        "prob":       probs,
    })
    return metrics, preds_df


def _get_ckpt_vocab_size(model_path, model_name):
    ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        return None
    key = "embedding.word_embeddings.weight"
    if key in state:
        return state[key].shape[0]
    for k, v in state.items():
        if "embedding" in k and "weight" in k and v.ndim == 2 and v.shape[-1] == D_MODEL:
            return v.shape[0]
    return None


def build_trainer(model_name, exp, seed, interaction, trainable_pt,
                  model_path, device):
    trainer = Trainer(seed=seed, device=device,
                      experiment_name=exp, verbose=False)

    ckpt_vocab = _get_ckpt_vocab_size(model_path, model_name)
    config = get_model_config(
        model_name=model_name, d_model=D_MODEL, n_layer=N_LAYER,
        verbose=False, rc=False,
        **({} if ckpt_vocab is None else {"vocab_size": ckpt_vocab}),
    )
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        config.trainable = trainable_pt

    trainer.define_model(
        config=config, model_name=model_name, pretrain=False,
        pooling_mode_target="mean", is_convblock=True,
        is_cross_attention=(interaction == "cross_attention"),
        interaction=interaction, use_unified_head=False,
        binding_pooling="mean", site_head_type="conv1d",
    )
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        if not trainable_pt:
            trainer.define_pretrained_model(model_name=model_name)

    trainer.set_pretrained_target(target="mirna", rna_model="rnabert")
    trainer.task = "sites"
    trainer.site_class_weights = None
    trainer.alpha = 0.5
    trainer.beta  = 0.5
    trainer.rc    = False
    trainer.load_model_from_path(str(model_path), verbose=False)
    return trainer


def run_experiment_seed(model_name, exp_tpl, interaction, trainable_pt,
                        df_train_raw, df_test_raw, seed, device, skip_preds):
    """단일 (exp_tpl, seed) 실행 → {split: metrics} dict"""
    exp        = f"{exp_tpl}_s{seed}"
    model_path = SAVED / model_name / exp / str(seed) / "train" / "model.pth"
    if not model_path.exists():
        print(f"  [SKIP] {exp} — checkpoint not found")
        return None

    pred_dir = PRED_DIR / exp
    pred_dir.mkdir(parents=True, exist_ok=True)

    # already done check
    splits_done = {
        s: (pred_dir / f"{s}_preds.pkl").exists()
        for s in ("train", "val", "test")
    }
    if not skip_preds and all(splits_done.values()):
        # preds 있으면 metrics만 재계산 (모델 로드 없이)
        print(f"  [CACHED] {exp} — loading from preds pkl")
        result = {}
        for split in ("train", "val", "test"):
            df_p = pd.read_pickle(pred_dir / f"{split}_preds.pkl")
            result[split] = compute_metrics(
                df_p["label"].to_numpy(), df_p["prob"].to_numpy()
            )
        return result

    print(f"  [RUN]  {exp}")

    ml = LM_MAX_LEN.get(model_name, MAX_LEN)
    train_ds, val_ds, test_ds = build_datasets(df_train_raw, df_test_raw, seed, ml)

    try:
        trainer = build_trainer(model_name, exp, seed, interaction,
                                trainable_pt, model_path, device)
    except Exception as e:
        print(f"  [ERROR] build_trainer {exp}: {e}")
        return None

    result = {}
    splits = [
        ("train", train_ds, 0, "Train"),
        ("val",   val_ds,   1, "Valid"),
        ("test",  test_ds,  2, "Test"),
    ]

    bs = LM_BS if model_name in ("rnamsm", "rnafm") else BS
    for split_name, ds, part, dtype in splits:
        trainer.set_dataloader(ds, part=part, batch_size=bs,
                               num_workers=WORKERS, shuffle=False)
        try:
            loader = (trainer.train_loader if part == 0
                      else trainer.valid_loader if part == 1
                      else trainer.test_loader)
            metrics, preds_df = run_split(trainer, loader, dtype)
        except Exception as e:
            print(f"  [ERROR] {exp} {split_name}: {e}")
            result[split_name] = None
            continue

        if metrics is None:
            result[split_name] = None
            continue

        result[split_name] = metrics

        if not skip_preds:
            preds_df.to_pickle(pred_dir / f"{split_name}_preds.pkl")

        print(f"    [{split_name:5s}] "
              f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
              f"F1mac={metrics['f1_macro']:.4f}  F1pos={metrics['f1_pos']:.4f}  "
              f"ACC={metrics['acc']:.4f}  MCC={metrics['mcc']:.4f}  "
              f"n={metrics['n_tokens']:,}  pos%={metrics['pos_rate']:.3f}")

    del trainer
    torch.cuda.empty_cache()
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
METRIC_KEYS = [
    "auroc", "auprc",
    "f1_pos", "prec_pos", "rec_pos",
    "f1_macro", "prec_macro", "rec_macro",
    "acc", "mcc", "threshold",
    "n_tokens", "n_pos", "n_neg", "pos_rate",
]

CSV_FIELDS = [
    "group", "label", "model_name", "exp_tpl", "seed", "split",
] + METRIC_KEYS


def save_group_csv(group, rows):
    """그룹별 독립 CSV에 저장 — 병렬 실행 시 충돌 없음."""
    csv_path = OUT / f"eval_full_{group}.csv"
    new_df = pd.DataFrame(rows, columns=CSV_FIELDS)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        key = ["exp_tpl", "seed", "split"]
        existing = existing[~existing.set_index(key).index.isin(
            new_df.set_index(key).index)]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    return csv_path


def merge_group_csvs():
    """eval_full_{group}.csv 들을 합쳐 eval_full_summary.csv 생성."""
    all_groups = ["encoder", "pretrained", "ablation",
                  "interaction", "site_head", "pretraining"]
    dfs = []
    for g in all_groups:
        p = OUT / f"eval_full_{g}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("[merge] 병합할 파일이 없음")
        return
    merged = pd.concat(dfs, ignore_index=True)
    out_path = OUT / "eval_full_summary.csv"
    merged.to_csv(out_path, index=False)
    print(f"[merge] {len(merged)} rows → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group", default="all",
                        help="all / encoder / pretrained / ablation / "
                             "interaction / site_head / pretraining / merge")
    parser.add_argument("--skip_preds", action="store_true",
                        help="metrics만 저장하고 preds pkl은 저장 안 함")
    args = parser.parse_args()

    # merge 모드: 그룹별 CSV를 합쳐서 summary 생성
    if args.group == "merge":
        merge_group_csvs()
        return

    groups = (
        ["encoder", "pretrained", "ablation", "interaction", "site_head", "pretraining"]
        if args.group == "all" else [args.group]
    )

    print("Loading raw data...")
    df_train_raw, df_test_raw = load_raw_data()
    print(f"  train={len(df_train_raw)}  test={len(df_test_raw)}")

    # 그룹별로 독립된 딕셔너리에 수집 → 그룹 CSV에 저장 (병렬 안전)
    group_rows: dict = {g: [] for g in groups}

    for group, label, model_name, exp_tpl, interaction, trainable in EXPERIMENTS:
        if group not in groups:
            continue

        print(f"\n[{group}] {label}")

        for seed in SEEDS:
            result = run_experiment_seed(
                model_name, exp_tpl, interaction, trainable,
                df_train_raw, df_test_raw, seed, args.device, args.skip_preds,
            )
            if result is None:
                continue

            for split, metrics in result.items():
                if metrics is None:
                    continue
                row = {
                    "group": group, "label": label,
                    "model_name": model_name, "exp_tpl": exp_tpl,
                    "seed": seed, "split": split,
                }
                for k in METRIC_KEYS:
                    row[k] = metrics.get(k, "")
                group_rows[group].append(row)

        # 실험 하나 끝날 때마다 해당 그룹 CSV에 저장 (중간 저장)
        if group_rows[group]:
            csv_path = save_group_csv(group, group_rows[group])
            group_rows[group] = []   # 다음 저장 시 중복 방지용 초기화
            print(f"  → {csv_path}")

    print(f"\n{'='*55}")
    for g in groups:
        p = OUT / f"eval_full_{g}.csv"
        if p.exists():
            print(f" {p.name}")
    print(f" merge: python scripts/eval_full.py --group merge")
    if not args.skip_preds:
        print(f" Preds → {PRED_DIR}/{{exp}}/{{train|val|test}}_preds.pkl")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
