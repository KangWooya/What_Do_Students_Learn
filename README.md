# What Do Students Learn? A Feature-Level Analysis of Dark Knowledge

**Seungu Kang, Songkuk Kim** — Yonsei University, Seoul, Republic of Korea  
*ICPR 2026*

---

## Overview

We analyze what knowledge distillation (KD) actually transfers at the **feature level** using the Interaction Tensor (IT) framework. Key findings:

1. **KD as regularization**: KD suppresses low-frequency, sample-specific features and reinforces high-frequency, reusable ones.
2. **Confusion = Dark Knowledge**: A teacher's confusion matrix encodes the same inter-class similarity structure as its soft targets (Pearson r ≈ 0.87, cosine ≈ 0.78).
3. **Confusion Distillation (CD)**: A teacher-free self-distillation method using the model's own confusion pattern as dynamic soft targets, outperforming CS-KD and PS-KD on CIFAR-100 by ~1.2%.

---

## Repository Structure

```
WDSL/
├── analysis/
│   ├── analyze_kd.py          # Figures 1 & 2 (IT analysis of KD)
│   ├── analyze_cd.py          # Figure 3 (IT analysis of CD)
│   ├── analyze_confusion.py   # Section 4.1 (confusion ≈ dark knowledge)
│   └── evaluate_models.py     # Table 2 (accuracy, mean ± std over 3 runs)
├── training/
│   ├── train.py               # Baseline training
│   ├── train_cd.py            # Confusion Distillation training
│   ├── train_kd.py            # Knowledge Distillation training
│   ├── models/                # ResNet, DenseNet definitions
│   └── conf/                  # Training configuration
├── baselines/
│   ├── cs-kd/                 # CS-KD (Yun et al., 2020)
│   └── ps-kd/                 # PS-KD (Kim et al., 2021)
├── checkpoints/               # Model weights — download from HuggingFace (see below)
├── interaction_tensors/       # Pre-computed IT tensors — download from HuggingFace
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
git clone https://github.com/<your-username>/WDSL.git
cd WDSL
uv sync          # Creates .venv and installs all dependencies
```

**CPU-only environment**: remove the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from `pyproject.toml` before running `uv sync`. Analysis of pre-computed tensors runs on CPU; model loading in `analyze_kd.py` and `evaluate_models.py` will be slower but functional.

**Other CUDA versions**: replace `cu124` with your version (e.g., `cu118`, `cu126`) in `pyproject.toml`.

---

## Data & Pre-computed Tensors

**Interaction Tensor files** (pre-computed, required for Figures 1–3) are hosted on HuggingFace:

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

**Model checkpoints** are not publicly released. Figures 1–3 and their statistics reproduce fully from the pre-computed tensors above. `analyze_confusion.py` and `evaluate_models.py` require locally trained checkpoints.

**CIFAR-100** is downloaded automatically on first run. Set `DATA_DIR` at the top of each script to your preferred download location.

---

## Reproducing Paper Results

All scripts are run from the repository root with `uv run`.

### Figure 1 & 2 — KD Feature Analysis (Section 3 & 4)

```bash
uv run python analysis/analyze_kd.py --out-dir figures/
# Options:
#   --data-dir PATH       CIFAR-100 root (default: /home/seungu/mycode/Data)
#   --out-dir PATH        save figures to directory (default: display only)
#   --skip-confidence     skip Fig 2a/b/c (requires loading 60 model checkpoints)
```

Requires: `interaction_tensors/interaction_tensor_KD_models_20teacher.pt`  
For Fig 2a/b/c also requires: `checkpoints/KDmodels/`, `checkpoints/Basemodels/`, `checkpoints/Teachers/`

### Figure 3 — CD Feature Analysis (Section 4.3)

```bash
uv run python analysis/analyze_cd.py --out-dir figures/
```

Requires: `interaction_tensors/interaction_tensor_B_SCD_model.pt`

### Section 4.1 — Confusion ≈ Dark Knowledge

```bash
uv run python analysis/analyze_confusion.py --out-dir figures/
# Options:
#   --teacher-ckpt PATH   path to ResNet-152 checkpoint
#   --baseline-ckpt PATH  path to ResNet-18 baseline checkpoint
```

Requires: `checkpoints/resnet/resnet152/Tuesday_18_November_2025_16h_22m_05s/resnet152-200-best.pth`  
and `checkpoints/Basemodels/bm1.pth`

### Table 2 — Accuracy Comparison

```bash
uv run python analysis/evaluate_models.py
# Option:
#   --data-dir PATH       CIFAR-100 root
```

Requires locally trained checkpoints in `checkpoints/resnet/`, `checkpoints/densenet121/`, `checkpoints/cs_kd/`, `checkpoints/ps_kd/`.

> **Note:** DenseNet-121 CD-200 checkpoints are not available; `evaluate_models.py` reproduces all other rows in Table 2 (ResNet-18/34/50 all methods, DenseNet-121 Baseline/CS-KD/PS-KD/CD-300).

---

## Reproducing Training

### Baseline & CD

```bash
cd training/

# Baseline ResNet-18
uv run python train.py -net resnet18

# Confusion Distillation ResNet-18 (300 epochs)
uv run python train_cd.py -net resnet18 \
    --transition_epoch 30 30 60 30 150 \
    --soft_w 0.7 --ce_w 0.3 --temperature 2.0 --ema_momentum 0.9
```

See `scripts/run_command.sh` for full commands including ResNet-34/50 and DenseNet-121.

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

To recompute IT tensors from scratch (instead of downloading pre-computed ones):

```bash
# 1. Train 20 baseline models and 20 KD student models (see scripts/run_command.sh)
# 2. Open analysis/1_kd_analysis.ipynb and run cells 1–37
#    (cells 28–37 build and save interaction_tensor_KD_models_20teacher.pt)
# 3. Open analysis/2_cd_it_analysis.ipynb and run cells 1–25
#    (builds interaction_tensor_B_SCD_model.pt)
```

The IT construction pipeline: hook activations from `conv5_x` → PCA(50) → correlation matrix → greedy clustering (Algorithm 1) → threshold → binary tensor Ω ∈ {0,1}^{M×N×T}.

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Temperature T | 2.0 | Soft label temperature |
| Loss weights | 0.7 : 0.3 | soft_w : ce_w during training |
| EMA momentum μ | 0.9 | Confusion matrix smoothing |
| Label smoothing | 0.1 | Initial smoothing matrix S |
| Phase schedule | 3:3:6:3:15 | CD phase alternation (×300 epochs) |

---

## Citation

```bibtex
@inproceedings{kang2026wdsl,
  title     = {What Do Students Learn? A Feature-Level Analysis of Dark Knowledge},
  author    = {Kang, Seungu and Kim, Songkuk},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```