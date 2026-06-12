"""
Build Interaction Tensors from trained checkpoints.

Implements the pipeline described in Section 3.2 of the paper:
  1. Hook intermediate-layer activations for M models over the test set
  2. PCA(50) per model  →  (M, N, 50)
  3. Pairwise cross-model correlation  →  Λ ∈ R^(M×M×K×K)
  4. Greedy cross-model clustering (Algorithm 1)  →  assignment (M, K)
  5. L∞-normalize and threshold activations  →  binary (M, N, K)
  6. Aggregate by cluster  →  Ω ∈ {0,1}^(M×N×T)
  7. Save to interaction_tensors/

Usage:
    # KD tensor  (20 KD students + 20 baselines + 20 teachers, hook: conv5_x)
    uv run python analysis/build_tensors.py --target kd

    # CD tensor  (20 baselines + 20 CD models, hook: avg_pool)
    uv run python analysis/build_tensors.py --target cd

Output:
    interaction_tensors/interaction_tensor_KD_models_20teacher.pt
    interaction_tensors/interaction_tensor_B_SCD_model.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── Path Configuration ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path("data/")   # set to your CIFAR-100 root
CKPT_DIR  = REPO_ROOT / "checkpoints"
IT_DIR    = REPO_ROOT / "interaction_tensors"
# ──────────────────────────────────────────────────────────────────────────


# ── ResNet definitions (CIFAR-100 variant, 32×32 input) ───────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, 1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
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
        x = self.conv2_x(x); x = self.conv3_x(x)
        x = self.conv4_x(x); x = self.conv5_x(x)
        x = self.avg_pool(x)
        return self.fc(x.view(x.size(0), -1))


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
        root=str(data_dir), train=False,
        download=not os.path.exists(os.path.join(str(data_dir), "cifar-100-python")),
        transform=transform,
    )
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ── Step 1: Collect activations via forward hook ──────────────────────────
def collect_activations(model, hook_layer, testloader, device):
    """Run model over testloader, collect flattened activations at hook_layer."""
    buf = []

    def hook_fn(module, inp, out):
        buf.append(torch.flatten(out, 1).detach().cpu())

    handle = hook_layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for images, _ in testloader:
            model(images.to(device))
    handle.remove()
    return torch.cat(buf, dim=0)   # (N, D)


# ── Step 2: PCA(50) per model ─────────────────────────────────────────────
def pca_project(pi_matrices, n_components=50):
    """
    pi_matrices: list of (N, D) CPU tensors, one per model.
    Returns: (M, N, 50) tensor.
    """
    proj_list = []
    for i, pi in enumerate(tqdm(pi_matrices, desc="PCA")):
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        proj = torch.from_numpy(pca.fit_transform(pi.numpy())).float()
        proj_list.append(proj.unsqueeze(0))
    return torch.cat(proj_list, dim=0)   # (M, N, 50)


# ── Step 3: Cross-model correlation tensor ────────────────────────────────
def compute_corr(proj_tensor):
    """
    proj_tensor: (M, N, K)
    Returns Λ: (M, M, K, K)
    """
    mean = proj_tensor.mean(dim=1, keepdim=True)
    std  = proj_tensor.std(dim=1, keepdim=True).clamp(min=1e-8)
    z = (proj_tensor - mean) / std                  # (M, N, K)
    z = z.permute(0, 2, 1)                          # (M, K, N)
    corr = torch.einsum("mkn,pqn->mpkq", z, z) / proj_tensor.shape[1]
    return corr


# ── Step 4: Greedy cross-model clustering (Algorithm 1) ───────────────────
def cluster_features(corr_tensor, gamma):
    """
    corr_tensor: (M, M, K, K) absolute correlations
    gamma: scalar threshold
    Returns assignment: (M, K) with cluster IDs 1..T
    """
    corr_abs = corr_tensor.abs()
    M, _, K, _ = corr_abs.shape
    assignment = torch.zeros(M, K, dtype=torch.long)
    maximum    = torch.full((M, K), -1.0)
    cluster_id = 1

    for i in tqdm(range(M), desc="Clustering"):
        for j in range(K):
            if assignment[i, j] != 0:
                continue
            assignment[i, j] = cluster_id
            for p in range(M):
                row = corr_abs[i, p, j, :]          # (K,)
                for q in range(K):
                    if row[q] > maximum[p, q] and row[q] > gamma:
                        assignment[p, q] = cluster_id
                        maximum[p, q]    = row[q]
            cluster_id += 1

    n_clusters = torch.unique(assignment).numel()
    print(f"Clusters: {n_clusters}")
    return assignment


# ── Step 5: Threshold activations → binary mask ───────────────────────────
def compute_data_feature(proj_tensor, threshold=0.9):
    """
    proj_tensor: (M, N, K)
    Returns binary tensor (M, N, K).
    """
    norm = proj_tensor.norm(p=float("inf"), dim=1, keepdim=True).clamp(min=1e-8)
    normalized = proj_tensor / norm
    r = torch.quantile(normalized.reshape(normalized.shape[0], -1), threshold, dim=1)
    return (normalized > r[:, None, None]).int()


# ── Step 6: Assemble Ω ────────────────────────────────────────────────────
def compute_interaction_tensor(assignment, data_feature):
    """
    assignment:   (M, K) — cluster IDs 1..T  (CPU)
    data_feature: (M, N, K) — binary          (CPU)
    Returns Ω: (M, N, T) binary int tensor.
    """
    M, N, K = data_feature.shape
    T = int(assignment.max().item())
    it = torch.zeros(M, N, T, dtype=torch.int32)
    idx = (assignment - 1).long()   # 0-indexed

    for t in tqdm(range(T), desc="Assembling IT"):
        mask = (idx == t).unsqueeze(1)              # (M, 1, K)
        it[:, :, t] = (mask * data_feature).sum(dim=2).bool().int()

    return it


# ── KD tensor pipeline ────────────────────────────────────────────────────
def build_kd_tensor(testloader, device):
    """
    60 models on the same feature axis (conv5_x):
      [0:20]  = KD students  (ResNet-18, checkpoints/KDmodels/kd{i}.pth)
      [20:40] = Baselines    (ResNet-18, checkpoints/Basemodels/bm{i}.pth)
      [40:60] = Teachers     (ResNet-152, checkpoints/Teachers/t{i}.pth)
    """
    pi_matrices = []

    configs = (
        [(f"{CKPT_DIR}/KDmodels/kd{i}.pth",   resnet18,  "conv5_x") for i in range(1, 21)] +
        [(f"{CKPT_DIR}/Basemodels/bm{i}.pth",  resnet18,  "conv5_x") for i in range(1, 21)] +
        [(f"{CKPT_DIR}/Teachers/t{i}.pth",     resnet152, "conv5_x") for i in range(1, 21)]
    )

    for ckpt_path, model_fn, layer_name in tqdm(configs, desc="Collecting activations"):
        model = model_fn().to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        hook_layer = getattr(model, layer_name)
        pi = collect_activations(model, hook_layer, testloader, device)
        pi_matrices.append(pi.cpu())
        del model

    return pi_matrices


# ── CD tensor pipeline ────────────────────────────────────────────────────
def build_cd_tensor(testloader, device):
    """
    40 models (avg_pool output, i.e., post-conv5_x):
      [0:20]  = Baselines  (ResNet-18, checkpoints/Basemodels/bm{i}.pth)
      [20:40] = CD models  (ResNet-18, checkpoints/CDmodels/cd{i}.pth)
    """
    pi_matrices = []

    configs = (
        [(f"{CKPT_DIR}/Basemodels/bm{i}.pth", "avg_pool") for i in range(1, 21)] +
        [(f"{CKPT_DIR}/CDmodels/cd{i}.pth",   "avg_pool") for i in range(1, 21)]
    )

    for ckpt_path, layer_name in tqdm(configs, desc="Collecting activations"):
        model = resnet18().to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        hook_layer = getattr(model, layer_name)
        pi = collect_activations(model, hook_layer, testloader, device)
        pi_matrices.append(pi.cpu())
        del model

    return pi_matrices


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build Interaction Tensor (Section 3.2)")
    parser.add_argument("--target", required=True, choices=["kd", "cd"],
                        help="kd: KD-student/baseline/teacher tensor  |  cd: baseline/CD-model tensor")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--corr-percentile", type=float, default=99.0,
                        help="Percentile of |correlation| used as clustering threshold γ")
    parser.add_argument("--data-threshold", type=float, default=0.9,
                        help="Per-model activation percentile for binary thresholding")
    args = parser.parse_args()

    IT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    testloader = make_testloader(args.data_dir)

    # Step 1 — collect activations
    if args.target == "kd":
        pi_matrices = build_kd_tensor(testloader, device)
        out_path = IT_DIR / "interaction_tensor_KD_models_20teacher.pt"
    else:
        pi_matrices = build_cd_tensor(testloader, device)
        out_path = IT_DIR / "interaction_tensor_B_SCD_model.pt"

    # Step 2 — PCA(50)
    print(f"\nProjecting {len(pi_matrices)} models to {args.pca_components} PCA components …")
    proj_tensor = pca_project(pi_matrices, n_components=args.pca_components)
    print(f"proj_tensor: {proj_tensor.shape}")   # (M, N, 50)

    # Step 3 — correlation
    print("\nComputing cross-model correlation tensor …")
    corr_tensor = compute_corr(proj_tensor)
    print(f"corr_tensor: {corr_tensor.shape}")   # (M, M, 50, 50)

    # Step 4 — clustering
    gamma = torch.quantile(corr_tensor.abs(), args.corr_percentile / 100.0).item()
    print(f"γ (p{args.corr_percentile:.0f}): {gamma:.4f}")
    assignment = cluster_features(corr_tensor, gamma)
    print(f"assignment: {assignment.shape}")     # (M, 50)

    # Step 5 — threshold
    print("\nThresholding activations …")
    data_feature = compute_data_feature(proj_tensor, threshold=args.data_threshold)
    print(f"data_feature: {data_feature.shape}") # (M, N, 50)

    # Step 6 — assemble Ω
    print("\nAssembling interaction tensor …")
    omega = compute_interaction_tensor(assignment, data_feature)
    print(f"Ω shape: {omega.shape}")             # (M, N, T)

    # Save
    torch.save(omega, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
