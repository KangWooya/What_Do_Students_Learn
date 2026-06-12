"""
Figures 1 & 2: Feature-level analysis of KD via Interaction Tensor.

Loads pre-computed interaction_tensor_KD_models_20teacher.pt (60 models:
  [0:20]  = KD student (ResNet-18)
  [20:40] = Baseline   (ResNet-18)
  [40:60] = Teacher    (ResNet-152)

Produces:
  Figure 1a — Feature frequency distribution (how many samples use each feature)
  Figure 1b — Common feature usage: for features shared by all three groups,
               how many samples each group uses them on (sorted)
  Figure 2a/b/c — Confidence vs. feature count KDE (student/baseline/teacher)
  Figure 2d — Feature count per sample (sorted)
  Section 3 stats — avg feature usage, avg data-feature count, unique counts
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# ── Path Configuration ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path("data/")   # set to your CIFAR-100 root
CKPT_DIR  = str(REPO_ROOT / "checkpoints")
IT_DIR    = str(REPO_ROOT / "interaction_tensors")
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


# ── Figure 1a: feature frequency distribution (full) ─────────────────────
def plot_feature_frequency_all(omega, out_path=None):
    model_slices = {
        "student":  omega[0:20],
        "baseline": omega[20:40],
        "teacher":  omega[40:60],
    }
    plt.figure(figsize=(10, 6))
    for name, part in model_slices.items():
        freq = ((part.sum(dim=0) > 0).int()).sum(dim=0)
        sorted_freq, _ = torch.sort(freq, descending=False)
        plt.plot(sorted_freq.cpu().numpy(), label=name)
    plt.title("Feature Frequency Distribution by Model Type")
    plt.xlabel("Feature (sorted)")
    plt.ylabel("Occurrences")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.show()


# ── Figure 1b: feature frequency distribution (zoomed, high-freq tail) ───
def plot_feature_frequency_zoom(omega, out_path=None):
    model_slices = {
        "student":  omega[0:20],
        "baseline": omega[20:40],
        "teacher":  omega[40:60],
    }
    plt.figure(figsize=(10, 6))
    freq_len = None
    for name, part in model_slices.items():
        freq = ((part.sum(dim=0) > 0).int()).sum(dim=0)
        sorted_freq, _ = torch.sort(freq, descending=False)
        freq_np = sorted_freq.cpu().numpy()
        freq_len = len(freq_np)
        plt.plot(np.arange(freq_len), freq_np, label=name)
    plt.title("Feature Frequency Distribution (High-Frequency Tail)")
    plt.xlabel("Feature (sorted)")
    plt.ylabel("Occurrences")
    plt.xlim(600, freq_len)
    plt.legend(); plt.grid(True); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.show()


# ── Figure 2a/b/c: Confidence vs. feature count KDE ──────────────────────
def get_confidence(interaction_tensor, model_name, testloader, device):
    conf_mat = []

    if model_name == "student":
        ckpt_fmt = f"{CKPT_DIR}/KDmodels/kd{{i}}.pth"
        model_fn = resnet18
        omega_slice = interaction_tensor[0:20]
    elif model_name == "baseline":
        ckpt_fmt = f"{CKPT_DIR}/Basemodels/bm{{i}}.pth"
        model_fn = resnet18
        omega_slice = interaction_tensor[20:40]
    elif model_name == "teacher":
        ckpt_fmt = f"{CKPT_DIR}/Teachers/t{{i}}.pth"
        model_fn = resnet152
        omega_slice = interaction_tensor[40:60]

    for model_id in range(1, 21):
        model = model_fn().to(device)
        model.load_state_dict(torch.load(ckpt_fmt.format(i=model_id), weights_only=True))
        model.eval()
        conf_list = []
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                conf = probs[torch.arange(probs.size(0)), labels]
                conf_list.append(conf.squeeze().detach())
        conf_mat.append(torch.cat(conf_list))

    conf_data = torch.stack(conf_mat).mean(dim=0)
    num_features = torch.sum(torch.sum(omega_slice, dim=0), dim=1)

    df = pd.DataFrame({
        "confidence":   conf_data.cpu().numpy(),
        "num_features": num_features.cpu().numpy(),
    })
    sns.kdeplot(data=df, x="confidence", y="num_features", fill=True, cmap="Blues", levels=10)
    plt.title(f"Confidence vs. Feature Count ({model_name})")
    plt.tight_layout()


def plot_confidence_all(interaction_tensor, testloader, device, out_dir=None):
    for name in ("student", "baseline", "teacher"):
        plt.figure(figsize=(7, 5))
        get_confidence(interaction_tensor, name, testloader, device)
        if out_dir:
            path = os.path.join(out_dir, f"fig2_{name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        plt.show()


# ── Figure 2d: feature count per sample (sorted) ─────────────────────────
def plot_data_feature_count_all(omega, out_path=None):
    model_slices = {
        "student":  omega[0:20],
        "baseline": omega[20:40],
        "teacher":  omega[40:60],
    }
    plt.figure(figsize=(10, 6))
    for name, part in model_slices.items():
        count = ((part.sum(dim=0) > 0).int()).sum(dim=1)
        sorted_count, _ = torch.sort(count, descending=False)
        plt.plot(sorted_count.cpu().numpy(), label=name)
    plt.title("Feature Count per Sample (sorted)")
    plt.xlabel("Sample (sorted)")
    plt.ylabel("Number of Features")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.show()


# ── Section 3 statistics ──────────────────────────────────────────────────
def count_used_features(omega):
    for name, part in [("student", omega[0:20]), ("baseline", omega[20:40]), ("teacher", omega[40:60])]:
        used = (part.sum(dim=(0, 1)) > 0).sum().item()
        print(f"{name}: {int(used)} unique features")


def avg_feature_usage(omega):
    for name, part in [("student", omega[0:20]), ("baseline", omega[20:40]), ("teacher", omega[40:60])]:
        freq = ((part.sum(dim=0) > 0).int()).sum(dim=0)
        used_mask = freq > 0
        unique   = int(used_mask.sum().item())
        total_occ = int(freq[used_mask].sum().item())
        print(f"{name}: {unique} features, {total_occ} total occurrences, avg {total_occ/unique:.2f} per feature")


def avg_data_feature_usage(omega):
    for name, part in [("student", omega[0:20]), ("baseline", omega[20:40]), ("teacher", omega[40:60])]:
        count = ((part.sum(dim=0) > 0).int()).sum(dim=1)
        print(f"{name}: avg {count.float().mean():.2f} features per sample")


# ── Additional: common feature usage across all three groups ───────────────
def plot_common_feature_usage(omega, out_path=None):
    student_part = omega[0:20]
    normal_part  = omega[20:40]
    teacher_part = omega[40:60]

    used_S = student_part.sum(dim=(0, 1)) > 0
    used_N = normal_part.sum(dim=(0, 1)) > 0
    used_T = teacher_part.sum(dim=(0, 1)) > 0

    common_idx = torch.where(used_S & used_N & used_T)[0]
    print(f"Common features (all three groups): {len(common_idx)}")

    freq_S = ((student_part.sum(dim=0) > 0).int())[:, common_idx].sum(dim=0)
    freq_N = ((normal_part.sum(dim=0) > 0).int())[:, common_idx].sum(dim=0)
    freq_T = ((teacher_part.sum(dim=0) > 0).int())[:, common_idx].sum(dim=0)

    plt.figure(figsize=(10, 6))
    for freq, label in [(freq_S, "student"), (freq_N, "baseline"), (freq_T, "teacher")]:
        sorted_f, _ = torch.sort(freq, descending=False)
        plt.plot(sorted_f.cpu().numpy(), label=label)
    plt.title("Usage of Common Features (per data count)")
    plt.xlabel("Common Features (sorted)")
    plt.ylabel("# of Samples where Feature Appears")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.show()
    print(f"  student avg: {freq_S.float().mean():.2f}")
    print(f"  baseline avg: {freq_N.float().mean():.2f}")
    print(f"  teacher avg: {freq_T.float().mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description="KD feature analysis (Figures 1 & 2)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Path to CIFAR-100 root (default: %(default)s)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to save figures (default: display only)")
    parser.add_argument("--skip-confidence", action="store_true",
                        help="Skip Figure 2a/b/c (requires loading 60 models)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.out_dir, name) if args.out_dir else None

    print("Loading interaction tensor …")
    it_path = os.path.join(IT_DIR, "interaction_tensor_KD_models_20teacher.pt")
    interaction_tensor = torch.load(it_path, weights_only=True)
    print(f"  shape: {interaction_tensor.shape}")  # (60, N, T)

    # ── Figure 1a
    print("\n[Figure 1a] Feature frequency distribution …")
    plot_feature_frequency_all(interaction_tensor, out("fig1a_feature_frequency.png"))

    # ── Figure 1b
    print("[Figure 1b] Common feature usage (features shared by all groups, sorted) …")
    plot_common_feature_usage(interaction_tensor, out("fig1b_common_feature_usage.png"))

    # ── Figure 2a/b/c
    if not args.skip_confidence:
        print("\n[Figure 2a/b/c] Confidence vs. feature count (loading 60 models) …")
        testloader = make_testloader(args.data_dir)
        plot_confidence_all(interaction_tensor, testloader, device, args.out_dir)

    # ── Figure 2d
    print("[Figure 2d] Feature count per sample …")
    plot_data_feature_count_all(interaction_tensor, out("fig2d_data_feature_count.png"))

    # ── Section 3 stats
    print("\n── Section 3 Statistics ──────────────────────────────────")
    print("[Unique features used]")
    count_used_features(interaction_tensor)
    print("\n[Average feature usage]")
    avg_feature_usage(interaction_tensor)
    print("\n[Average features per sample]")
    avg_data_feature_usage(interaction_tensor)


if __name__ == "__main__":
    main()