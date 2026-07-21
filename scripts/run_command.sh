#!/bin/bash
# Training commands for all methods × architectures (CIFAR-100)
#
# Run from the repository root:
#   bash scripts/run_command.sh
#
# Or copy individual sections and run manually.
#
# DATA_DIR: set to your CIFAR-100 root (default: data/)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-data/}"

TRAIN="$REPO_ROOT/training"
CSKD="$REPO_ROOT/baselines/cs-kd"
PSKD="$REPO_ROOT/baselines/ps-kd"

# ──────────────────────────────────────────────────────────────
# Baseline (200 epoch)
# conf/global_settings.py: EPOCH=200, MILESTONES=[60,120,160]
# ──────────────────────────────────────────────────────────────
cd "$TRAIN"
uv run python train.py -net resnet18
uv run python train.py -net resnet34
uv run python train.py -net resnet50
uv run python train.py -net densenet121

# ──────────────────────────────────────────────────────────────
# Knowledge Distillation — ResNet-18 student, ResNet-152 teacher
# conf/global_settings.py: EPOCH=200, MILESTONES=[60,120,160]
# ──────────────────────────────────────────────────────────────
cd "$TRAIN"
uv run python train_kd.py \
    -net resnet18 -gpu \
    --kd \
    --teacher_net resnet152 \
    --teacher_path <path-to-resnet152-best.pth> \
    --kd_T 2.0 --kd_alpha 0.85

# ──────────────────────────────────────────────────────────────
# Confusion Distillation — 200 epoch
# Phase schedule 3:3:3:3:8 × (200/20 = 10) → 30 30 30 30 80
# conf/global_settings.py: EPOCH=200, MILESTONES=[60,120,160]
# ──────────────────────────────────────────────────────────────
cd "$TRAIN"
uv run python train_cd.py -net resnet18 -gpu --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net resnet34 -gpu --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net resnet50 -gpu --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net densenet121 -gpu --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

# ──────────────────────────────────────────────────────────────
# Confusion Distillation — 300 epoch (best config, Table 1)
# Phase schedule 3:3:6:3:15 × (300/30 = 10) → 30 30 60 30 150
# conf/global_settings.py: EPOCH=300, MILESTONES=[90,180,240]
# ──────────────────────────────────────────────────────────────
cd "$TRAIN"
uv run python train_cd.py -net resnet18 -gpu --confkd \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net resnet34 -gpu --confkd \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net resnet50 -gpu --confkd \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

uv run python train_cd.py -net densenet121 -gpu --confkd \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

# ──────────────────────────────────────────────────────────────
# CS-KD (200 epoch)
# ──────────────────────────────────────────────────────────────
cd "$CSKD"
uv run python train.py --dataset cifar100 --model CIFAR_ResNet18 \
    --dataroot "$DATA_DIR" --sgpu 0 --lr 0.1 --epoch 200 \
    --name run1 --decay 1e-4 -cls --lamda 1

uv run python train.py --dataset cifar100 --model CIFAR_ResNet34 \
    --dataroot "$DATA_DIR" --sgpu 0 --lr 0.1 --epoch 200 \
    --name run1 --decay 1e-4 -cls --lamda 1

uv run python train.py --dataset cifar100 --model CIFAR_ResNet50 \
    --dataroot "$DATA_DIR" --sgpu 0 --lr 0.1 --epoch 200 \
    --name run1 --decay 1e-4 -cls --lamda 1

uv run python train.py --dataset cifar100 --model CIFAR_DenseNet121 \
    --dataroot "$DATA_DIR" --sgpu 0 --lr 0.1 --epoch 200 \
    --name run1 --decay 1e-4 -cls --lamda 1

# ──────────────────────────────────────────────────────────────
# PS-KD (300 epoch, lr decay at 150 and 225)
# ──────────────────────────────────────────────────────────────
cd "$PSKD"
uv run python main.py --data_type cifar100 --data_path "$DATA_DIR" \
    --classifier_type ResNet18 --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 --workers 4 \
    --experiments_dir ./runs/pskd_resnet18_cifar100

uv run python main.py --data_type cifar100 --data_path "$DATA_DIR" \
    --classifier_type ResNet34 --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 --workers 4 \
    --experiments_dir ./runs/pskd_resnet34_cifar100

uv run python main.py --data_type cifar100 --data_path "$DATA_DIR" \
    --classifier_type ResNet50 --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 --workers 4 \
    --experiments_dir ./runs/pskd_resnet50_cifar100

uv run python main.py --data_type cifar100 --data_path "$DATA_DIR" \
    --classifier_type DenseNet121 --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 --workers 4 \
    --experiments_dir ./runs/pskd_densenet121_cifar100