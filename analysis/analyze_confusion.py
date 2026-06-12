"""
Section 4.1: Confusion matrix ≈ Dark Knowledge.

Loads a ResNet-152 teacher and a ResNet-18 baseline from checkpoints,
then computes per-class cosine similarity, Pearson r, Spearman r, and
soft IoU between the teacher's average softmax outputs and the baseline's
confusion matrix (both with self-prediction excluded).

Key results reported in the paper:
  Pearson r ≈ 0.87,  cosine ≈ 0.78
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# ── Path Configuration ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path("data/")   # set to your CIFAR-100 root
CKPT_DIR  = str(REPO_ROOT / "checkpoints")

# Set these to your own checkpoints
TEACHER_CKPT  = ""   # e.g. "checkpoints/resnet/resnet152/<run>/resnet152-200-best.pth"
BASELINE_CKPT = ""   # e.g. "checkpoints/Basemodels/bm1.pth"
# ──────────────────────────────────────────────────────────────────────────


# ── ResNet definitions (CIFAR-100 variant, 32×32 input) ───────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_x = self._make_layer(block, 64,  num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18():  return ResNet(BasicBlock, [2, 2, 2, 2])
def resnet152(): return ResNet(BottleNeck, [3, 8, 36, 3])
# ──────────────────────────────────────────────────────────────────────────


def make_testloader(data_dir, batch_size=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883,  0.27615047132568404),
        ),
    ])
    testset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=not os.path.exists(os.path.join(str(data_dir), "cifar-100-python")), transform=transform
    )
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def get_avg_softmax(model, testloader, device, num_classes=100):
    """Compute per-class average softmax output over the test set."""
    sum_probs = np.zeros((num_classes, num_classes), dtype=float)
    count = np.zeros(num_classes, dtype=int)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for j in range(num_classes):
                mask = labels_np == j
                if mask.any():
                    sum_probs[j] += probs[mask].sum(axis=0)
                    count[j] += mask.sum()
    return sum_probs / count[:, None]


def get_confusion_matrix(model, testloader, device, num_classes=100):
    """Compute confusion matrix over the test set."""
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            for t, p in zip(labels.numpy(), preds):
                conf_mat[t, p] += 1
    return conf_mat


def soft_iou(P, Q, eps=1e-12):
    inter = np.minimum(P, Q).sum()
    union = np.maximum(P, Q).sum()
    return inter / union if union >= eps else 0.0


def compute_metrics(avg_softmax, conf_mat, num_classes=100):
    """
    Per-class cosine, Pearson r, and soft IoU between teacher avg softmax
    and baseline confusion probability (self-prediction excluded).
    """
    conf_prob = (conf_mat.astype(float) + 1e-8) / conf_mat.sum(axis=1, keepdims=True)

    cos_per_class = []
    iou_per_class = []

    for j in range(num_classes):
        P = np.delete(avg_softmax[j], j)
        Q = np.delete(conf_prob[j], j)
        cos_per_class.append(cosine_similarity(P.reshape(1, -1), Q.reshape(1, -1))[0, 0])
        iou_per_class.append(soft_iou(P, Q))

    # Global Pearson / Spearman (diagonal excluded)
    diag_idx = np.arange(0, num_classes * num_classes, num_classes + 1)
    flat_sf = np.delete(avg_softmax.flatten(), diag_idx)
    flat_cp = np.delete(conf_prob.flatten(), diag_idx)

    pearson_r = np.corrcoef(flat_sf, flat_cp)[0, 1]
    spearman_r = spearmanr(flat_sf, flat_cp).correlation

    return {
        "cosine_per_class": np.array(cos_per_class),
        "iou_per_class":    np.array(iou_per_class),
        "pearson_r":        pearson_r,
        "spearman_r":       spearman_r,
    }


def print_and_plot_metrics(metrics, out_dir=None):
    cos_arr = metrics["cosine_per_class"]
    iou_arr = metrics["iou_per_class"]

    print(f"\n── Section 4.1 Metrics ───────────────────────────────────")
    print(f"Pearson r  (self excluded): {metrics['pearson_r']:.4f}")
    print(f"Spearman r (self excluded): {metrics['spearman_r']:.4f}")
    print(f"Mean cosine similarity:     {cos_arr.mean():.4f} ± {cos_arr.std():.4f}")
    print(f"Mean soft IoU:              {iou_arr.mean():.4f} ± {iou_arr.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(cos_arr, kde=True, bins=20, color="tab:green", ax=axes[0])
    axes[0].set_title("Cosine Similarity (per class, self excluded)")
    axes[0].set_xlabel("Cosine similarity")

    sns.histplot(iou_arr, kde=True, bins=20, color="tab:orange", ax=axes[1])
    axes[1].set_title("Soft IoU (per class, self excluded)")
    axes[1].set_xlabel("IoU")

    plt.tight_layout()
    if out_dir:
        path = os.path.join(out_dir, "fig_confusion_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Confusion ≈ Dark Knowledge (Section 4.1)")
    parser.add_argument("--data-dir",      default=str(DATA_DIR))
    parser.add_argument("--teacher-ckpt",  default=TEACHER_CKPT)
    parser.add_argument("--baseline-ckpt", default=BASELINE_CKPT)
    parser.add_argument("--out-dir",       default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    testloader = make_testloader(args.data_dir)

    print("Loading teacher (ResNet-152) …")
    teacher = resnet152().to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, weights_only=True))

    print("Computing teacher average softmax …")
    avg_softmax = get_avg_softmax(teacher, testloader, device)

    print("Loading baseline (ResNet-18) …")
    baseline = resnet18().to(device)
    baseline.load_state_dict(torch.load(args.baseline_ckpt, weights_only=True))

    print("Computing confusion matrix …")
    conf_mat = get_confusion_matrix(baseline, testloader, device)

    metrics = compute_metrics(avg_softmax, conf_mat)
    print_and_plot_metrics(metrics, args.out_dir)


if __name__ == "__main__":
    main()