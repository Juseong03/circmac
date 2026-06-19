#!/usr/bin/env python3
"""
check_progress.py — 실험 진행 현황 확인 스크립트

실험별로 done / running / pending 상태를 표시합니다.
  DONE    : saved_models/{model}/{exp}_s{seed}/{seed}/train/model.pth 존재
  RUNNING : 로그 파일이 최근 N분 이내 수정됨 (기본 10분)
  PENDING : 위 두 조건 모두 미충족

Usage:
    python scripts/check_progress.py
    python scripts/check_progress.py --group encoder
    python scripts/check_progress.py --recent 5       # running 판단 기준 분
    python scripts/check_progress.py --seeds 1 2 3    # 특정 seed만 확인
    python scripts/check_progress.py --verbose        # 미완료 실험 목록 출력
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "saved_models"
LOGS  = ROOT / "logs" / "v2"

SEEDS_ALL = list(range(1, 11))

# ── 실험 정의: (group, label, model_name, exp_template, log_subdir) ─────────
# exp_template: 실제 디렉토리 = saved_models/{model}/{template}_s{seed}/{seed}/train/model.pth
# log_subdir  : logs/v2/{log_subdir}/{template}_s{seed}.log  (None이면 탐색 안 함)

EXPERIMENTS = [
    # ── Encoder comparison ─────────────────────────────────────────────────
    ("encoder", "LSTM",        "lstm",        "v2_enc_lstm",        "enc"),
    ("encoder", "Transformer", "transformer", "v2_enc_transformer", "enc"),
    ("encoder", "Mamba",       "mamba",       "v2_enc_mamba",       "enc"),
    ("encoder", "Hymba",       "hymba",       "v2_enc_hymba",       "enc"),
    ("encoder", "CircMAC",     "circmac",     "v2_abl_full",        "abl"),

    # ── Ablation ──────────────────────────────────────────────────────────
    ("ablation", "CircMAC (full)",  "circmac", "v2_abl_full",         "abl"),
    ("ablation", "Attn only",       "circmac", "v2_abl_attn_only",    "abl"),
    ("ablation", "Mamba only",      "circmac", "v2_abl_mamba_only",   "abl"),
    ("ablation", "CNN only",        "circmac", "v2_abl_cnn_only",     "abl"),
    ("ablation", "No Attn",         "circmac", "v2_abl_no_attn",      "abl"),
    ("ablation", "No Mamba",        "circmac", "v2_abl_no_mamba",     "abl"),
    ("ablation", "No Conv",         "circmac", "v2_abl_no_conv",      "abl"),
    ("ablation", "No CircBias",     "circmac", "v2_abl_no_circ_bias", "abl"),

    # ── Interaction mechanism ─────────────────────────────────────────────
    ("interaction", "Concat",      "circmac", "v2_int_concat",      "int"),
    ("interaction", "Elementwise", "circmac", "v2_int_elementwise", "int"),
    ("interaction", "Cross-Attn",  "circmac", "v2_int_cross_attn",  "int"),

    # ── Site head ─────────────────────────────────────────────────────────
    ("site_head", "Conv1D", "circmac", "v2_head_conv1d", "head"),
    ("site_head", "Linear", "circmac", "v2_head_linear", "head"),

    # ── Pretraining strategy (finetune) ───────────────────────────────────
    ("pretraining", "NoPT",          "circmac", "v2_pt_nopt",          "pt"),
    ("pretraining", "MLM",           "circmac", "v2_pt_mlm",           "pt"),
    ("pretraining", "NTP",           "circmac", "v2_pt_ntp",           "pt"),
    ("pretraining", "SSP",           "circmac", "v2_pt_ssp",           "pt"),
    ("pretraining", "CPCL",          "circmac", "v2_pt_cpcl",          "pt"),
    ("pretraining", "BSJ",           "circmac", "v2_pt_bsj",           "pt"),
    ("pretraining", "MLM+NTP",       "circmac", "v2_pt_mlm_ntp",       "pt"),
    ("pretraining", "MLM+SSP",       "circmac", "v2_pt_mlm_ssp",       "pt"),
    ("pretraining", "MLM+CPCL",      "circmac", "v2_pt_mlm_cpcl",      "pt"),
    ("pretraining", "Pairing",       "circmac", "v2_pt_pairing",       "pt"),
    ("pretraining", "MLM+CPCL+SSP",  "circmac", "v2_pt_mlm_cpcl_ssp",  "pt"),
    ("pretraining", "All",           "circmac", "v2_pt_all",           "pt"),

    # ── RNA-LM Frozen ─────────────────────────────────────────────────────
    ("rna_frozen", "RNABERT (frozen)",  "rnabert",  "exp1_fair_frozen_rnabert",  "rna_frozen"),
    ("rna_frozen", "RNAErnie (frozen)", "rnaernie", "exp1_fair_frozen_rnaernie", "rna_frozen"),
    ("rna_frozen", "RNA-MSM (frozen)",  "rnamsm",   "exp1_fair_frozen_rnamsm",   "rna_frozen"),
    ("rna_frozen", "RNA-FM (frozen)",   "rnafm",    "exp1_fair_frozen_rnafm",    "rna_frozen"),

    # ── RNA-LM Trainable ──────────────────────────────────────────────────
    ("rna_trainable", "RNABERT (ft)",  "rnabert",  "exp1_fair_trainable_rnabert",  "rna_trainable"),
    ("rna_trainable", "RNAErnie (ft)", "rnaernie", "exp1_fair_trainable_rnaernie", "rna_trainable"),
    ("rna_trainable", "RNA-MSM (ft)",  "rnamsm",   "exp1_fair_trainable_rnamsm",   "rna_trainable"),
    ("rna_trainable", "RNA-FM (ft)",   "rnafm",    "exp1_fair_trainable_rnafm",    "rna_trainable"),
]

# ── Pretraining checkpoints (model.pth, seed=42) ────────────────────────────
PRETRAIN_STRATEGIES = [
    ("mlm",          "--mlm"),
    ("ntp",          "--ntp"),
    ("mlm_ssp",      "--mlm --ssp"),
    ("mlm_cpcl",     "--mlm --cpcl"),
    ("mlm_ntp",      "--mlm --ntp"),
    ("all",          "--mlm --ntp --ssp --pairing --cpcl --bsj_mlm"),
    ("ssp",          "--ssp"),
    ("cpcl",         "--cpcl"),
    ("bsj",          "--bsj_mlm"),
    ("pairing",      "--pairing"),
    ("mlm_cpcl_ssp", "--mlm --cpcl --ssp"),
]


def model_done(model_name: str, exp_template: str, seed: int) -> bool:
    """saved_models/{model}/{template}_s{seed}/{seed}/train/model.pth 존재 여부"""
    p = SAVED / model_name / f"{exp_template}_s{seed}" / str(seed) / "train" / "model.pth"
    return p.exists()


def log_running(log_subdir: str, exp_template: str, seed: int, recent_min: int) -> bool:
    """로그 파일이 recent_min 분 이내 수정되었으면 True"""
    if not log_subdir:
        return False
    log_path = LOGS / log_subdir / f"{exp_template}_s{seed}.log"
    if not log_path.exists():
        return False
    age_sec = time.time() - log_path.stat().st_mtime
    return age_sec <= recent_min * 60


def pretrain_done(strategy: str) -> bool:
    p = SAVED / "circmac" / f"v2_ptm_{strategy}" / "42" / "pretrain" / "model.pth"
    return p.exists()


def bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done:>3}/{total}"


GROUPS_ORDER = [
    "encoder", "ablation", "interaction", "site_head",
    "pretraining", "rna_frozen", "rna_trainable",
]

GROUP_LABELS = {
    "encoder":      "EXP1  Encoder Comparison",
    "ablation":     "EXP4  CircMAC Ablation",
    "interaction":  "EXP5  Interaction Mechanism",
    "site_head":    "EXP6  Site Head",
    "pretraining":  "EXP2  Pretraining Strategy",
    "rna_frozen":   "EXP1  RNA-LM Frozen",
    "rna_trainable":"EXP1  RNA-LM Trainable",
}


def main():
    parser = argparse.ArgumentParser(description="실험 진행 현황 확인")
    parser.add_argument("--group", nargs="*", choices=list(GROUP_LABELS.keys()),
                        help="특정 그룹만 확인 (기본: 전체)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="확인할 seed 목록 (기본: 1-10)")
    parser.add_argument("--recent", type=int, default=10,
                        help="RUNNING 판단 기준: 로그 최근 수정 분 (기본: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="미완료 실험 목록 출력")
    args = parser.parse_args()

    seeds = args.seeds or SEEDS_ALL
    target_groups = set(args.group) if args.group else set(GROUP_LABELS.keys())

    # ── 그룹별 집계 ──────────────────────────────────────────────────────────
    # key: (group, label, template) → 중복 제거 (e.g., v2_abl_full이 encoder/ablation 양쪽에 등장)
    seen_templates = set()
    group_data: dict[str, list] = {g: [] for g in GROUPS_ORDER}

    for group, label, model, template, log_sub in EXPERIMENTS:
        if group not in target_groups:
            continue
        key = (group, template)
        if key in seen_templates:
            continue
        seen_templates.add(key)

        seed_status = {}  # seed -> "done" | "running" | "pending"
        for s in seeds:
            if model_done(model, template, s):
                seed_status[s] = "done"
            elif log_running(log_sub, template, s, args.recent):
                seed_status[s] = "running"
            else:
                seed_status[s] = "pending"

        group_data[group].append({
            "label":    label,
            "model":    model,
            "template": template,
            "status":   seed_status,
        })

    # ── 화면 출력 ─────────────────────────────────────────────────────────────
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    total_done = total_run = total_pend = 0

    print()
    print("=" * 70)
    print(f"  실험 진행 현황   (seeds: {seeds[0]}–{seeds[-1]},  recent={args.recent}min)")
    print(f"  {now_str}")
    print("=" * 70)

    # ── Pretraining checkpoint 상태 ───────────────────────────────────────
    if "pretraining" in target_groups:
        pt_done = sum(1 for s, _ in PRETRAIN_STRATEGIES if pretrain_done(s))
        pt_total = len(PRETRAIN_STRATEGIES)
        print()
        print(f"  [Pretrain checkpoints]  {bar(pt_done, pt_total)}  ({pt_done}/{pt_total} 완료)")
        if args.verbose or pt_done < pt_total:
            for strat, flags in PRETRAIN_STRATEGIES:
                mark = "✓" if pretrain_done(strat) else "✗"
                print(f"    {mark}  v2_ptm_{strat:<20}  {flags}")

    # ── 그룹별 현황 ───────────────────────────────────────────────────────
    for group in GROUPS_ORDER:
        if group not in target_groups:
            continue
        rows = group_data[group]
        if not rows:
            continue

        n_seeds = len(seeds)
        g_done = g_run = g_pend = 0
        for r in rows:
            for s in seeds:
                st = r["status"].get(s, "pending")
                if st == "done":    g_done += 1
                elif st == "running": g_run += 1
                else:               g_pend += 1

        g_total = len(rows) * n_seeds
        total_done += g_done; total_run += g_run; total_pend += g_pend

        print()
        print(f"  ── {GROUP_LABELS[group]} ──")
        print(f"     진행률: {bar(g_done, g_total)}  "
              f"(done={g_done}, running={g_run}, pending={g_pend})")

        # 각 실험별 seed 현황
        seed_header = "  ".join(f"s{s:02d}" for s in seeds)
        print(f"     {'모델':<28}  {seed_header}")
        print(f"     {'-'*28}  {'-'*5*n_seeds}")
        for r in rows:
            marks = []
            for s in seeds:
                st = r["status"].get(s, "pending")
                if st == "done":      marks.append(" ✓ ")
                elif st == "running": marks.append("[→]")
                else:                 marks.append(" · ")
            print(f"     {r['label']:<28}  {'  '.join(marks)}")

            if args.verbose:
                pending_seeds = [s for s in seeds if r["status"].get(s) != "done"]
                if pending_seeds:
                    print(f"       └─ 미완료 seeds: {pending_seeds}")

    # ── 전체 요약 ─────────────────────────────────────────────────────────
    grand_total = total_done + total_run + total_pend
    print()
    print("=" * 70)
    print(f"  전체 요약: {bar(total_done, grand_total, 30)}")
    print(f"    done={total_done}  running={total_run}  pending={total_pend}  "
          f"total={grand_total}")
    print("=" * 70)
    print()

    # ── 현재 실행 중인 Python 프로세스 ────────────────────────────────────
    print("  [현재 실행 중인 프로세스]")
    result = os.popen("ps aux | grep 'python training\\|python pretraining' | grep -v grep").read().strip()
    if result:
        for line in result.splitlines():
            parts = line.split()
            pid  = parts[1]
            cpu  = parts[2]
            mem  = parts[3]
            cmd  = " ".join(parts[10:])
            # exp name 추출
            exp_name = ""
            if "--exp" in cmd:
                idx = cmd.split().index("--exp")
                exp_name = cmd.split()[idx + 1] if idx + 1 < len(cmd.split()) else ""
            gpu_val = ""
            if "--device" in cmd:
                idx = cmd.split().index("--device")
                gpu_val = "GPU" + cmd.split()[idx + 1] if idx + 1 < len(cmd.split()) else ""
            print(f"    PID={pid:<7} {gpu_val:<5} CPU={cpu}%  MEM={mem}%  exp={exp_name}")
    else:
        print("    (실행 중인 실험 없음)")
    print()


if __name__ == "__main__":
    main()
