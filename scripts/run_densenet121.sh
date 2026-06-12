#!/bin/bash
# DenseNet-121 × 4방법 × 3회 학습 스크립트
# Generated: 2026-04-06
# Methods: Baseline (200ep), CS-KD (200ep), PS-KD (300ep), CD (200ep, schedule 30+30+60+30+150)

BASE=/home/seungu/mycode/WDSL/Confusion_distillation
DATA=/home/seungu/mycode/Data
LOG_DIR="$BASE/logs_densenet121"
STATUS_FILE="$LOG_DIR/STATUS.txt"

mkdir -p "$LOG_DIR"

log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_FILE"
}

log_status "=== DenseNet-121 학습 시작 (4 methods × 3 runs) ==="
log_status "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ─────────────────────────────────────────────
# 1. Baseline (200 epoch) × 3회
# ─────────────────────────────────────────────
cd "$BASE/pytorch-cifar100"
for i in 1 2 3; do
    log_status "[Baseline] Run $i/3 시작"
    python train.py -net densenet121 -gpu \
        > "$LOG_DIR/baseline_run${i}.log" 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        LAST_ACC=$(grep "Test set: Accuracy" "$LOG_DIR/baseline_run${i}.log" | tail -1)
        log_status "[Baseline] Run $i/3 완료 ✓  $LAST_ACC"
    else
        log_status "[Baseline] Run $i/3 실패 (exit=$EXIT_CODE)"
    fi
done

# ─────────────────────────────────────────────
# 2. CS-KD (200 epoch) × 3회
# ─────────────────────────────────────────────
cd "$BASE/cs-kd"
for i in 1 2 3; do
    log_status "[CS-KD] Run $i/3 시작"
    python train.py \
        --sgpu 0 --lr 0.1 --epoch 200 \
        --model CIFAR_DenseNet121 \
        --name densenet121_run${i} \
        --decay 1e-4 --dataset cifar100 \
        --dataroot "$DATA" \
        -cls --lamda 1 \
        > "$LOG_DIR/cskd_run${i}.log" 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        LAST_ACC=$(grep -E "acc|Acc|best" "$LOG_DIR/cskd_run${i}.log" | tail -3)
        log_status "[CS-KD] Run $i/3 완료 ✓"
        log_status "  $LAST_ACC"
    else
        log_status "[CS-KD] Run $i/3 실패 (exit=$EXIT_CODE)"
    fi
done

# ─────────────────────────────────────────────
# 3. PS-KD (300 epoch) × 3회
# ─────────────────────────────────────────────
cd "$BASE/PS-KD-Pytorch"
for i in 1 2 3; do
    log_status "[PS-KD] Run $i/3 시작"
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --lr 0.1 --lr_decay_schedule 150 225 \
        --PSKD --alpha_T 0.8 \
        --experiments_dir ./runs/pskd_densenet121_cifar100 \
        --batch_size 128 \
        --classifier_type DenseNet121 \
        --data_path "$DATA" \
        --data_type cifar100 \
        --workers 4 \
        > "$LOG_DIR/pskd_run${i}.log" 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        LAST_ACC=$(grep -E "acc|Acc|best|Top" "$LOG_DIR/pskd_run${i}.log" | tail -3)
        log_status "[PS-KD] Run $i/3 완료 ✓"
        log_status "  $LAST_ACC"
    else
        log_status "[PS-KD] Run $i/3 실패 (exit=$EXIT_CODE)"
    fi
done

# ─────────────────────────────────────────────
# 4. CD (200 epoch) × 3회
# ─────────────────────────────────────────────
cd "$BASE/pytorch-cifar100"
for i in 1 2 3; do
    log_status "[CD] Run $i/3 시작"
    python train_cd.py \
        -net densenet121 -gpu -b 128 \
        --confkd \
        --T 2.0 --soft_w 0.7 --ce_w 0.3 \
        --smoothing 0.1 \
        --transition_epoch 30 30 60 30 150 \
        --ema 0.9 \
        > "$LOG_DIR/cd_run${i}.log" 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        LAST_ACC=$(grep "Test set: Accuracy" "$LOG_DIR/cd_run${i}.log" | tail -1)
        log_status "[CD] Run $i/3 완료 ✓  $LAST_ACC"
    else
        log_status "[CD] Run $i/3 실패 (exit=$EXIT_CODE)"
    fi
done

log_status "=== 전체 완료 ==="
