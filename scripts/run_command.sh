#!/bin/bash
# Linux 환경 학습 명령어 (원본: run_command.txt, Windows PowerShell 기준)
# 작업 디렉토리: 각 명령의 cd 참고
# 데이터 경로: /home/seungu/mycode/Data

BASE=/home/seungu/mycode/WDSL/Confusion_distillation
DATA=/home/seungu/mycode/Data

# ============================================================
# Baseline ResNet-18 (pytorch-cifar100)
# ============================================================
cd "$BASE/pytorch-cifar100"
python train.py -net resnet18 -gpu

# ============================================================
# Teacher ResNet-152 x 20회 순차 학습
# ============================================================
cd "$BASE/pytorch-cifar100"
for i in $(seq 1 20); do
    echo "===== Run $i / 20 ====="
    python train.py -net resnet152 -gpu
done

# ============================================================
# Confusion Distillation (CD) - ResNet-18, 300epoch
# 전환 epoch 비율: 3:3:6:3:15 = 30 30 60 30 150 (300 epoch 기준)
# ============================================================
cd "$BASE/pytorch-cifar100"
python train_cd.py \
    -net resnet18 \
    -gpu \
    -b 128 \
    --confkd \
    --T 2.0 \
    --soft_w 0.7 \
    --ce_w 0.3 \
    --smoothing 0.1 \
    --transition_epoch "30 30 60 30 150" \
    --ema 0.9

# ============================================================
# CS-KD - ResNet-18 (200 epoch)
# ============================================================
cd "$BASE/cs-kd"
python train.py \
    --sgpu 0 \
    --lr 0.1 \
    --epoch 200 \
    --model CIFAR_ResNet18 \
    --name test_cifar \
    --decay 1e-4 \
    --dataset cifar100 \
    --dataroot "$DATA" \
    -cls \
    --lamda 1

# CS-KD - ResNet-34
python train.py \
    --sgpu 0 \
    --lr 0.1 \
    --epoch 200 \
    --model CIFAR_ResNet34 \
    --name test_cifar \
    --decay 1e-4 \
    --dataset cifar100 \
    --dataroot "$DATA" \
    -cls \
    --lamda 1

# CS-KD - ResNet-50
python train.py \
    --sgpu 0 \
    --lr 0.1 \
    --epoch 200 \
    --model CIFAR_ResNet50 \
    --name test_cifar \
    --decay 1e-4 \
    --dataset cifar100 \
    --dataroot "$DATA" \
    -cls \
    --lamda 1

# ============================================================
# PS-KD - ResNet-18
# ============================================================
cd "$BASE/PS-KD-Pytorch"
CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr 0.1 \
    --lr_decay_schedule 150 225 \
    --PSKD \
    --experiments_dir ./runs/pskd_resnet18_cifar100 \
    --batch_size 128 \
    --classifier_type ResNet18 \
    --data_path "$DATA" \
    --data_type cifar100 \
    --alpha_T 0.8 \
    --workers 4

# PS-KD - ResNet-34
CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr 0.1 \
    --lr_decay_schedule 150 225 \
    --PSKD \
    --experiments_dir ./runs/pskd_resnet34_cifar100 \
    --batch_size 128 \
    --classifier_type ResNet34 \
    --data_path "$DATA" \
    --data_type cifar100 \
    --alpha_T 0.8 \
    --workers 4

# ============================================================
# [카메라 레디] DenseNet-121 on CIFAR-100
# ============================================================

# Baseline DenseNet-121
cd "$BASE/pytorch-cifar100"
python train.py -net densenet121 -gpu

# CD DenseNet-121
python train_cd.py \
    -net densenet121 \
    -gpu \
    -b 128 \
    --confkd \
    --T 2.0 \
    --soft_w 0.7 \
    --ce_w 0.3 \
    --smoothing 0.1 \
    --transition_epoch "30 30 60 30 150" \
    --ema 0.9

# CS-KD DenseNet-121
cd "$BASE/cs-kd"
python train.py \
    --sgpu 0 \
    --lr 0.1 \
    --epoch 200 \
    --model CIFAR_DenseNet121 \
    --name test_cifar \
    --decay 1e-4 \
    --dataset cifar100 \
    --dataroot "$DATA" \
    -cls \
    --lamda 1

# PS-KD DenseNet-121
cd "$BASE/PS-KD-Pytorch"
CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr 0.1 \
    --lr_decay_schedule 150 225 \
    --PSKD \
    --experiments_dir ./runs/pskd_densenet121_cifar100 \
    --batch_size 128 \
    --classifier_type DenseNet121 \
    --data_path "$DATA" \
    --data_type cifar100 \
    --alpha_T 0.8 \
    --workers 4

# PS-KD - ResNet-18 resume (기존 체크포인트 이어서 학습)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_type cifar100 \
    --data_path "$DATA" \
    --classifier_type ResNet18 \
    --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 \
    --experiments_dir ./runs/pskd_resnet34_cifar100 \
    --resume "$BASE/PS-KD-Pytorch/runs/pskd_resnet18_cifar100/cifar100_ResNet18_PSKD_True_2025-9-4-15-59-56/model/checkpoint_best.pth"
