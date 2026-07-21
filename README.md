# What Do Students Learn? A Feature-Level Analysis of Dark Knowledge

**Seungu Kang, Songkuk Kim** — Yonsei University, Seoul, Republic of Korea  
*Accepted at ICPR 2026 (to appear)* · [arXiv:2606.03052](https://arxiv.org/abs/2606.03052)

---

## Overview

We analyze how Knowledge Distillation (KD) changes student feature representations using the **Interaction Tensor** (IT) framework. Key findings and contributions:

1. **KD as feature-level regularization**: KD prunes low-frequency, sample-specific features and promotes a compact set of highly reusable ones. Student models learn 250 fewer features than baselines (531 → 281) while achieving higher average activation frequencies (1329 → 2121 per feature).

2. **Confusion = Dark Knowledge**: The dataset-level confusion matrix encodes inter-class similarity analogous to the teacher's soft targets (Pearson r ≈ 0.87, cosine ≈ 0.78). The confusion ratio does not replicate fine-grained probability values but captures a coarse inter-class similarity structure.

3. **Confusion Distillation (CD)**: A teacher-free self-distillation method that uses the model's own EMA-smoothed confusion matrix as dynamic soft targets. On ResNet-34 and ResNet-50 for CIFAR-100, CD outperforms CS-KD and PS-KD by ~1.2%.

---

## Repository Structure

```
WDSL/
├── analysis/
│   ├── analyze_kd.py          # Figures 1 & 2 — IT analysis of KD (Section 3)
│   ├── analyze_cd.py          # Figure 3 — IT analysis of CD (Section 4.3)
│   ├── analyze_confusion.py   # Section 4.1 — Confusion ≈ Dark Knowledge
│   └── evaluate_models.py     # Table 2 — accuracy, mean ± std over 3 runs
├── training/
│   ├── train.py               # Baseline training
│   ├── train_cd.py            # Confusion Distillation training
│   ├── train_kd.py            # Knowledge Distillation training
│   ├── models/                # ResNet, DenseNet definitions (CIFAR-100 variant)
│   └── conf/global_settings.py
├── baselines/
│   ├── cs-kd/                 # CS-KD (Yun et al., CVPR 2020)
│   └── ps-kd/                 # PS-KD (Kim et al., ICCV 2021)
├── scripts/
│   └── run_command.sh         # Training command examples
├── pyproject.toml
└── uv.lock
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management with PyTorch CUDA 12.4.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/KangWooya/What_Do_Students_Learn.git
cd What_Do_Students_Learn
uv sync          # Creates .venv and installs all dependencies
```

**CPU-only environment**: remove the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from `pyproject.toml` before running `uv sync`.

**Other CUDA versions**: replace `cu124` with your version (e.g., `cu118`, `cu126`) in `pyproject.toml`.

---

## Data & Pre-computed Tensors

**Interaction Tensor files** (required for Figures 1–3) are hosted on HuggingFace:

```bash
pip install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='KangWooya/WDSL-tensors',
                  repo_type='dataset',
                  local_dir='interaction_tensors/')
"
```

| File | Size | Used by |
|---|---|---|
| `interaction_tensor_KD_models_20teacher.pt` | ~1.8 GB | `analyze_kd.py` (Figs 1 & 2) |
| `interaction_tensor_B_SCD_model.pt` | ~896 MB | `analyze_cd.py` (Fig 3) |

Each tensor encodes Ω ∈ {0,1}^{M×N×T} — binary indicator of which model uses which feature on which sample:
- `interaction_tensor_KD_models_20teacher.pt` shape `[60, 10000, T]`: `[0:20]` = KD students, `[20:40]` = baselines, `[40:60]` = teachers
- `interaction_tensor_B_SCD_model.pt` shape `[40, 10000, T]`: `[0:20]` = baselines, `[20:40]` = CD models

**Model checkpoints** are not publicly released. Figures 1–3 reproduce fully from the pre-computed tensors. `analyze_confusion.py` and `evaluate_models.py` require locally trained checkpoints (see Training below).

**CIFAR-100** is downloaded automatically on first run. Set `DATA_DIR` at the top of each script to your preferred download location.

---

## Reproducing Paper Results

All scripts are run from the repository root with `uv run`.

### Figures 1 & 2 — KD Feature Analysis (Section 3)

```bash
uv run python analysis/analyze_kd.py --out-dir figures/
# Options:
#   --data-dir PATH       CIFAR-100 root (default: data/)
#   --out-dir PATH        save figures (default: display only)
#   --skip-confidence     skip Fig 2a/b/c (requires 60 model checkpoints)
```

Produces: `fig1a_feature_frequency.png`, `fig1b_common_feature_usage.png`, `fig2a/b/c` (confidence vs. feature count KDE), `fig2d_data_feature_count.png`.  
Requires: `interaction_tensors/interaction_tensor_KD_models_20teacher.pt`  
For Fig 2a/b/c also requires: `checkpoints/KDmodels/`, `checkpoints/Basemodels/`, `checkpoints/Teachers/`

### Figure 3 — CD Feature Analysis (Section 4.3)

```bash
uv run python analysis/analyze_cd.py --out-dir figures/
```

Produces: `fig3a_feature_frequency.png`, `fig3b_data_feature_count.png`.  
Requires: `interaction_tensors/interaction_tensor_B_SCD_model.pt`

### Section 4.1 — Confusion ≈ Dark Knowledge

```bash
uv run python analysis/analyze_confusion.py --out-dir figures/ \
    --teacher-ckpt  checkpoints/<your-resnet152-run>/resnet152-200-best.pth \
    --baseline-ckpt checkpoints/Basemodels/bm1.pth
```

Computes per-class cosine similarity, Pearson r, Spearman r, and soft IoU between the teacher's class-wise average softmax and the baseline's confusion ratio (diagonal excluded).  
Expected output: Pearson r ≈ 0.85, mean cosine ≈ 0.76, mean Jaccard ≈ 0.38.

### Table 2 — Accuracy Comparison

```bash
uv run python analysis/evaluate_models.py --data-dir data/
```

Fill in the checkpoint paths at the top of each method block in [analysis/evaluate_models.py](analysis/evaluate_models.py). Empty lists are skipped with a `(skipped)` message.

---

## Reproducing Training

### Baseline

```bash
cd training/
uv run python train.py -net resnet18      # or resnet34, resnet50, densenet121
```

The epoch count and LR milestones are set in [training/conf/global_settings.py](training/conf/global_settings.py):

```python
EPOCH = 200
MILESTONES = [60, 120, 160]
```

### Knowledge Distillation (Section 3)

```bash
cd training/
uv run python train_kd.py -net resnet18 -gpu \
    --kd \
    --teacher_net resnet152 \
    --teacher_path <path-to-resnet152-best.pth> \
    --kd_T 2.0 --kd_alpha 0.85
```

### Confusion Distillation (Section 4)

CD alternates between hard-target (CE only) and soft-target (confusion + CE) phases. The `--transition_epoch` argument specifies the absolute epoch at which each phase boundary occurs.

**Phase schedule for 200 epochs** — ratio 3:3:3:3:8, boundaries at every 10 epochs:

```
Epoch   0– 30  Hard
Epoch  30– 60  Soft
Epoch  60– 90  Hard
Epoch  90–120  Soft
Epoch 120–200  Hard (stable)
```

```bash
# conf/global_settings.py: EPOCH=200, MILESTONES=[60,120,160]
cd training/
uv run python train_cd.py -net resnet18 -gpu \
    --confkd \
    --transition_epoch 30 30 30 30 80 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9
```

**Phase schedule for 300 epochs** — ratio 3:3:6:3:15, boundaries at every 10 epochs (best config, Table 1):

```
Epoch   0– 30  Hard
Epoch  30– 60  Soft
Epoch  60–120  Hard
Epoch 120–150  Soft
Epoch 150–300  Hard (stable)
```

```bash
# conf/global_settings.py: EPOCH=300, MILESTONES=[90,180,240]
cd training/
uv run python train_cd.py -net resnet18 -gpu \
    --confkd \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --T 2.0 --ema 0.9
```

Replace `-net resnet18` with `resnet34`, `resnet50`, or `densenet121` as needed. See [scripts/run_command.sh](scripts/run_command.sh) for all architectures.

### CS-KD Baseline

```bash
cd baselines/cs-kd/
uv run python train.py --dataset cifar100 --model CIFAR_ResNet18 \
    --dataroot /path/to/data --sgpu 0 --lr 0.1 --epoch 200 \
    --name run1 --decay 1e-4 -cls --lamda 1
# model options: CIFAR_ResNet18 / CIFAR_ResNet34 / CIFAR_ResNet50 / CIFAR_DenseNet121
```

### PS-KD Baseline

```bash
cd baselines/ps-kd/
uv run python main.py --data_type cifar100 --data_path /path/to/data \
    --classifier_type ResNet18 --batch_size 128 \
    --lr 0.1 --lr_decay_schedule 150 225 \
    --PSKD --alpha_T 0.8 --workers 4 \
    --experiments_dir ./runs/pskd_resnet18_cifar100
# classifier_type options: ResNet18 / ResNet34 / ResNet50 / DenseNet121
```

---

## Interaction Tensor Construction

The Interaction Tensor framework is based on Jiang et al. (2023) — [arXiv:2306.04793](https://arxiv.org/abs/2306.04793). We adapt it for cross-model feature comparison across knowledge distillation methods.

To recompute IT tensors from trained checkpoints, use the standalone script:

```bash
# KD tensor  (20 KD students + 20 baselines + 20 teachers  →  shape [60, N, T])
# Requires: checkpoints/KDmodels/kd{1..20}.pth
#           checkpoints/Basemodels/bm{1..20}.pth
#           checkpoints/Teachers/t{1..20}.pth
uv run python analysis/build_tensors.py --target kd --data-dir data/

# CD tensor  (20 baselines + 20 CD models  →  shape [40, N, T])
# Requires: checkpoints/Basemodels/bm{1..20}.pth
#           checkpoints/CDmodels/cd{1..20}.pth
uv run python analysis/build_tensors.py --target cd --data-dir data/
```

The script implements the full pipeline from Section 3.2:
1. Hook intermediate-layer activations (conv5_x for KD, avg_pool for CD)
2. PCA(50) per model
3. Cross-model correlation tensor Λ
4. Greedy clustering (Algorithm 1) with 99th-percentile threshold γ
5. L∞-normalize and threshold activations → binary (M, N, K)
6. Aggregate by cluster → Ω ∈ {0,1}^(M×N×T)

The source notebooks (`analysis/1_kd_analysis.ipynb`, `analysis/2_cd_it_analysis.ipynb`) contain additional exploratory analysis and visualization cells.

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Temperature T | 2.0 | Soft label temperature (KD and CD) |
| KD loss ratio (soft:hard) | 0.85 : 0.15 | Used when training KD students (Section 3.1) |
| CD loss ratio (soft:hard) | 0.7 : 0.3 | Best CD configuration (Table 1) |
| EMA momentum μ | 0.9 | Confusion matrix smoothing (Eq. 6) |
| Label smoothing ε | 0.1 | Initial smoothing matrix S (Eq. 7) |
| Phase schedule | 3:3:6:3:15 | CD phase alternation × 300 epochs (Table 1) |
| Weight decay | 5×10⁻⁴ | SGD optimizer |

The phase schedule `3:3:6:3:15` over 300 epochs means:
```
Epoch   0– 30: Hard (CE only)
Epoch  30– 60: Soft (CD + CE)
Epoch  60–120: Hard
Epoch 120–150: Soft
Epoch 150–300: Hard (stable)
```

---

## Main Results (Table 2, CIFAR-100)

| Method | Epochs | ResNet-18 | ResNet-34 | ResNet-50 | DenseNet-121 |
|---|---|---|---|---|---|
| Baseline | 200 | 75.86±.09 | 77.61±.35 | 78.48±.56 | 79.03±.11 |
| CS-KD | 200 | 76.38±.17 | 76.73±.06 | 76.31±.36 | 76.53±.58 |
| PS-KD | 300 | 77.41±.22 | 77.33±.12 | 78.41±.31 | 79.84±.23 |
| CD (Ours) | 200 | 76.85±.10 | 77.87±.08 | 78.63±.21 | 78.71±.18 |
| **CD (Ours)** | **300** | **77.13±.01** | **78.53±.22** | **79.38±.23** | 79.64±.22 |

Top-1 accuracy, mean ± std over 3 runs. Bold = best among self-distillation methods.

---

## Acknowledgments

This repository builds on the following open-source codebases:

- **pytorch-cifar100** (baseline and CD training): [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- **CS-KD** (baseline comparison): [alinlab/cs-kd](https://github.com/alinlab/cs-kd)
- **PS-KD-Pytorch** (baseline comparison): [lgcnsai/PS-KD-Pytorch](https://github.com/lgcnsai/PS-KD-Pytorch)

The Interaction Tensor framework used for feature-level analysis is based on:

> Jiang et al., "On the Joint Interaction of Models, Data, and Features," arXiv:2306.04793 (2023)

This work was supported in part by IITP grants (No. RS-2024-00395824, No. RS-2025-02214652) funded by the Korea Government (MSIT).

---

## Citation

```bibtex
@inproceedings{kang2026wdsl,
  title     = {What Do Students Learn? {A} Feature-Level Analysis of Dark Knowledge},
  author    = {Kang, Seungu and Kim, Songkuk},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  note      = {arXiv:2606.03052}
}
```