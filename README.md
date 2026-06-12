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

> **Note**: DenseNet-121 CD-200 checkpoints are not available in this release; all other rows in Table 2 are reproducible.

---

## Reproducing Training

### Baseline

```bash
cd training/
uv run python train.py -net resnet18      # or resnet34, resnet50, densenet121
```

### Knowledge Distillation (Section 3)

```bash
cd training/
uv run python train_kd.py -net resnet18 \
    --teacher_net resnet152 \
    --teacher_weights <path-to-teacher.pth> \
    --temperature 2 --alpha 0.85
```

### Confusion Distillation (Section 4, best config)

```bash
cd training/
uv run python train_cd.py -net resnet18 \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --temperature 2.0 --ema_momentum 0.9
```

See [scripts/run_command.sh](scripts/run_command.sh) for commands for all architectures (ResNet-34/50, DenseNet-121).

### CS-KD Baseline

```bash
cd baselines/cs-kd/
uv run python train.py --dataset cifar100 --model CIFAR_ResNet18 \
    --data-dir /path/to/data
```

### PS-KD Baseline

```bash
cd baselines/ps-kd/
uv run python main.py --dataset cifar100 --classifier_type ResNet18 \
    --data_path /path/to/data --lr_decay_schedule 150 225
```

---

## Interaction Tensor Construction

To recompute IT tensors from trained checkpoints:

```
1. Train 20 independent baseline models + 20 KD student models + 20 teacher models
2. Open analysis/1_kd_analysis.ipynb — run cells that compute the IT:
   - Hook conv5_x layer activations
   - PCA(50) per model
   - Greedy cross-model correlation clustering (Algorithm 1 in paper)
   - Threshold → binary Ω ∈ {0,1}^{M×N×T}
   - Saves interaction_tensor_KD_models_20teacher.pt
3. For the CD tensor, open analysis/2_cd_it_analysis.ipynb
```

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