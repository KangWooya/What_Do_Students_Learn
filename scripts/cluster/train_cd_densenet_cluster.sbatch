#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Yonsei AI Data Center — DenseNet-121 Confusion Distillation, 200 epochs
# Reproduces the (previously lost) Table 2 "CD (200 ep) / DenseNet-121" row.
# 3 independent runs via a Slurm array (run1/run2/run3 → mean ± std).
#
# Submit from the repo root on a login node:
#   cd /home/kangswoo/WDSL && mkdir -p logs && sbatch scripts/cluster/train_cd_densenet_cluster.sbatch
#
# Phase schedule 3:3:3:3:8 → --transition_epoch 30 30 30 30 80  (Hard,Soft,Hard,Soft,Hard)
# ─────────────────────────────────────────────────────────────────────────
#SBATCH -J wdsl_cd_dn121
#SBATCH -o /home/kangswoo/WDSL/logs/%x.%A_%a.out
#SBATCH -e /home/kangswoo/WDSL/logs/%x.%A_%a.err
#SBATCH -p base_suma_rtx3090
#SBATCH --qos base_qos
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=1-3

set -e
source ~/.bashrc
module load cuda/12.8 2>/dev/null || true   # torch wheels are self-contained; harmless if unavailable

REPO=/home/kangswoo/WDSL
SCRATCH=/scratch/kangswoo/WDSL

# CD-200ep config + isolated per-run checkpoint dir (avoids array-task collisions)
export WDSL_EPOCH=200
export WDSL_DATA="$SCRATCH/data"
export WDSL_CKPT="$SCRATCH/checkpoint/run${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WDSL_CKPT"

echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname) | data=$WDSL_DATA | ckpt=$WDSL_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/training"
uv run python -u train_cd.py -net densenet121 -gpu --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9

echo "Task ${SLURM_ARRAY_TASK_ID} done. Best checkpoints under $WDSL_CKPT/densenet121/"
