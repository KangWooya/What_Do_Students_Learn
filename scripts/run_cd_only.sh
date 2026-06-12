#!/bin/bash
BASE=/home/seungu/mycode/WDSL/Confusion_distillation
LOG_DIR="$BASE/logs_densenet121"
STATUS_FILE="$LOG_DIR/STATUS.txt"

log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_FILE"
}

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
log_status "[CD] 전체 완료"
