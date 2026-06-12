"""
Build Interaction Tensors from trained checkpoints.

Implements the pipeline from Section 3.2 of the paper, directly
derived from the analysis notebooks (1_kd_analysis.ipynb Cells 28–37,
2_cd_it_analysis.ipynb Cells 18–24). Based on the Interaction Tensor
framework of Ge et al. (2023) — arXiv:2306.04793.

Pipeline (both targets):
  1. Hook activations for M models over the test set  →  PI_matrices
  2. PCA(50) per model                                →  PI_proj_tensor (M, N, 50)
  3. Cross-model correlation                          →  corr_tensor (M, M, K, K)
  4. Greedy clustering (Algorithm 1)                  →  assignment (M, K)
  5. L∞-normalize + threshold                         →  data_feature (M, N, K)
  6. Aggregate by cluster                             →  interaction_tensor Ω (M, N, T)

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


# ══ Step 1 ── Collect activations via forward hook ════════════════════════
# Notebook 1 Cell 28 (KD):  make_PI_with_hook — conv5_x, M=60
# Notebook 2 Cell 18 (CD):  make_PI_with_hook — avg_pool, M=40
#
# Returns PI_matrices: dict {"PI_matrix_m{i}": Tensor(N, D)} on CPU

def make_PI_with_hook_kd(testloader, device):
    """
    60 models: [1–20] KD students, [21–40] baselines, [41–60] teachers.
    Hook: conv5_x → (N, 8192) for ResNet-18, (N, 32768) for ResNet-152.
    """
    PI_matrices = {}
    for model_id in tqdm(range(1, 61), desc="Activations"):
        if model_id <= 20:
            model = resnet18().to(device)
            model.load_state_dict(torch.load(f'{CKPT_DIR}/KDmodels/kd{model_id}.pth', weights_only=True))
        elif model_id <= 40:
            student_id = model_id - 20
            model = resnet18().to(device)
            model.load_state_dict(torch.load(f'{CKPT_DIR}/Basemodels/bm{student_id}.pth', weights_only=True))
        else:
            teacher_id = model_id - 40
            model = resnet152().to(device)
            model.load_state_dict(torch.load(f'{CKPT_DIR}/Teachers/t{teacher_id}.pth', weights_only=True))

        model.eval()
        PI_list = []

        def hook_fn(module, input, output):
            flat_output = torch.flatten(output, 1)  # (batch, D)
            PI_list.append(flat_output.detach())

        handle = model.conv5_x.register_forward_hook(hook_fn)
        with torch.no_grad():
            for images, _ in testloader:
                model(images.to(device))

        PI_matrix = torch.cat(PI_list, dim=0)  # (N, D)
        PI_matrices[f'PI_matrix_m{model_id}'] = PI_matrix.cpu()
        handle.remove()
        del model

    return PI_matrices


def make_PI_with_hook_cd(testloader, device):
    """
    40 models: [1–20] baselines, [21–40] CD models.
    Hook: avg_pool → (N, 512) for ResNet-18.
    """
    PI_matrices = {}
    for model_id in tqdm(range(1, 41), desc="Activations"):
        model = resnet18().to(device)
        if model_id <= 20:
            model.load_state_dict(torch.load(f'{CKPT_DIR}/Basemodels/bm{model_id}.pth', weights_only=True))
        else:
            model.load_state_dict(torch.load(f'{CKPT_DIR}/CDmodels/cd{model_id - 20}.pth', weights_only=True))

        model.eval()
        PI_list = []

        def hook_fn(module, input, output):
            flat_output = torch.flatten(output, 1)  # (batch, 512)
            PI_list.append(flat_output.detach())

        handle = model.avg_pool.register_forward_hook(hook_fn)
        with torch.no_grad():
            for images, _ in testloader:
                model(images.to(device))

        PI_matrix = torch.cat(PI_list, dim=0)  # (N, 512)
        PI_matrices[f'PI_matrix_m{model_id}'] = PI_matrix.cpu()
        handle.remove()
        del model

    return PI_matrices


# ══ Step 2 ── PCA(50) per model ═══════════════════════════════════════════
# Notebook 1 Cell 30 (KD):  sklearn PCA — randomized SVD with mean centering
# Notebook 2 Cell 19 (CD):  torch.svd   — no mean centering
#
# The two notebooks use different PCA methods; both reduce to (N, 50).
# Returns PI_proj_tensor: (M, N, 50) CPU tensor.

def make_PI_proj_kd(PI_matrices, num_models):
    """Notebook 1 Cell 30: sklearn PCA(50, randomized, seed=42) with mean centering."""
    PI_proj_list = []
    for model_id in tqdm(range(1, num_models + 1), desc="PCA"):
        PI_matrix = PI_matrices[f'PI_matrix_m{model_id}']   # (N, D)
        PI_numpy = PI_matrix.detach().cpu().numpy()
        pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
        PI_proj_numpy = pca.fit_transform(PI_numpy)          # (N, 50)
        PI_proj = torch.from_numpy(PI_proj_numpy).float()
        PI_proj_list.append(PI_proj.unsqueeze(0))
        print(f"Model {model_id}: PI_proj shape = {PI_proj.shape}")

    PI_proj_tensor = torch.cat(PI_proj_list, dim=0)          # (M, N, 50)
    print(f"PI_proj_tensor = {PI_proj_tensor.shape}")
    return PI_proj_tensor


def make_PI_proj_cd(PI_matrices, num_models):
    """Notebook 2 Cell 19: manual torch.svd without mean centering."""
    PI_proj_list = []
    for model_id in tqdm(range(1, num_models + 1), desc="SVD"):
        PI_matrix = PI_matrices[f'PI_matrix_m{model_id}']   # (N, D)
        _, S, V = torch.svd(PI_matrix)
        sorted_indices = torch.argsort(S, descending=True)
        V_sorted = V[:, sorted_indices]
        V_pca50 = V_sorted[:, :50]                           # (D, 50)
        PI_proj = PI_matrix @ V_pca50                        # (N, 50)
        PI_proj_list.append(PI_proj.unsqueeze(0))
        print(f"Model {model_id}: PI_proj shape = {PI_proj.shape}")

    PI_proj_tensor = torch.cat(PI_proj_list, dim=0)          # (M, N, 50)
    print(f"PI_proj_tensor = {PI_proj_tensor.shape}")
    return PI_proj_tensor


# ══ Step 3 ── Cross-model correlation tensor ══════════════════════════════
# Notebook 1 Cell 32, Notebook 2 Cell 20: compute_corr

def compute_corr(proj_matrix):
    """
    proj_matrix: (M, N, K)
    Returns corr_tensor Λ: (M, M, K, K)
    """
    mean = torch.mean(proj_matrix, dim=1, keepdim=True)
    std  = torch.std(proj_matrix, dim=1, keepdim=True)
    proj_normalized = (proj_matrix - mean) / std           # (M, N, K)
    proj_reshape = proj_normalized.permute(0, 2, 1)        # (M, K, N)
    corr_tensor = torch.einsum('mkn,pqn->mpkq', proj_reshape, proj_reshape) / proj_matrix.shape[1]
    return corr_tensor


# Notebook 1 Cell 33, Notebook 2 Cell 20: get_gamma_corr

def get_gamma_corr(corr_tensor, percentile=99):
    corr_abs = torch.abs(corr_tensor)
    gamma_corr = torch.quantile(corr_abs, percentile / 100.0)
    print(gamma_corr)
    return gamma_corr


# ══ Step 4 ── Greedy cross-model clustering (Algorithm 1) ════════════════
# Notebook 1 Cell 34, Notebook 2 Cell 21: cluster_features
#
# Note on `maximum` dtype: both notebooks use dtype=torch.int32, which
# truncates float correlations to integers (0 for any value in [0, 1)).
# This means the "maximum seen" tracking degrades after the first assignment:
# once maximum[p,q] is set to 0, any corr > gamma (which is > 0) satisfies
# the > maximum condition. In practice the gamma threshold becomes the sole
# gate. Kept as-is for exact reproducibility with the pre-computed tensors.

def cluster_features(corr_tensor, gamma_corr):
    """
    corr_tensor: (M, M, K, K)
    gamma_corr:  scalar threshold (γ, 99th-percentile of |Λ|)
    Returns assignment: (M, K) with cluster IDs 1..T
    """
    corr_tensor = torch.abs(corr_tensor)
    M, _, K, _ = corr_tensor.shape
    assignment = torch.zeros((M, K))
    maximum = torch.full((M, K), -1.0, dtype=torch.int32)
    currentFeature = 1

    for i in tqdm(range(M), desc="Clustering"):
        for j in range(K):
            if assignment[i, j] != 0:
                continue
            assignment[i, j] = currentFeature
            for p in range(M):
                CorrMat = corr_tensor[i, p, :, :]
                FeatureRow = CorrMat[j, :]
                for q in range(K):
                    if (FeatureRow[q] > maximum[p, q]) & (FeatureRow[q] > gamma_corr):
                        assignment[p, q] = currentFeature
                        maximum[p, q] = FeatureRow[q]
            currentFeature += 1

    cluster_num = torch.unique(assignment.flatten())
    assert len(cluster_num) == torch.max(cluster_num)
    print(f'Num of clusters: {len(cluster_num)}')
    return assignment


# ══ Step 5 ── Threshold activations → binary mask ════════════════════════
# Notebook 1 Cell 35, Notebook 2 Cell 22: compute_data_feature

def compute_data_feature(proj_matrix, data_threshold=0.9):
    """
    proj_matrix: (M, N, K)
    Returns data_feature: (M, N, K) bool tensor.
    """
    proj_matrix_norm = torch.norm(proj_matrix, p=float('inf'), dim=1, keepdim=True)
    normalized_proj_matrix = proj_matrix / proj_matrix_norm
    r_data = torch.quantile(
        normalized_proj_matrix.view(normalized_proj_matrix.shape[0], -1),
        data_threshold, dim=1,
    )
    data_feature = normalized_proj_matrix > r_data[:, None, None]
    return data_feature


# ══ Step 6 ── Assemble Ω ═════════════════════════════════════════════════
# Notebook 1 Cell 36, Notebook 2 Cell 23: compute_interaction_tensor

def compute_interaction_tensor(assign_mat, data_feature):
    """
    assign_mat:   (M, K) — cluster IDs 1..T
    data_feature: (M, N, K) — bool
    Returns interaction_tensor Ω: (M, N, T) int32
    """
    M = data_feature.shape[0]
    N = data_feature.shape[1]
    T = len(torch.unique(assign_mat.flatten()))

    interaction_tensor = torch.zeros((M, N, T), dtype=torch.int64)
    assign_mask = assign_mat - 1   # 0-indexed

    for t in tqdm(range(interaction_tensor.shape[2]), desc="Assembling IT"):
        mask = (assign_mask == t).unsqueeze(1)
        interaction_tensor[:, :, t] = torch.sum(mask * data_feature, dim=2)

    interaction_tensor = interaction_tensor.bool().int()
    return interaction_tensor


# ── Main ──────────────────────────────────────────────────────────────────
# Notebook 1 Cell 37, Notebook 2 Cell 24: main pipeline

def main():
    parser = argparse.ArgumentParser(description="Build Interaction Tensor (Section 3.2)")
    parser.add_argument("--target", required=True, choices=["kd", "cd"],
                        help="kd: KD-student/baseline/teacher  |  cd: baseline/CD-model")
    parser.add_argument("--data-dir",        default=str(DATA_DIR))
    parser.add_argument("--corr-percentile", type=float, default=99.0,
                        help="Percentile for γ (clustering threshold)")
    parser.add_argument("--data-threshold",  type=float, default=0.9,
                        help="Per-model activation threshold for binary mask")
    args = parser.parse_args()

    IT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    testloader = make_testloader(args.data_dir)

    # Step 1 + 2 — activations + PCA
    if args.target == "kd":
        PI_matrices    = make_PI_with_hook_kd(testloader, device)     # Cell 28
        PI_proj_tensor = make_PI_proj_kd(PI_matrices, num_models=60)  # Cell 30
        out_path = IT_DIR / "interaction_tensor_KD_models_20teacher.pt"
    else:
        PI_matrices    = make_PI_with_hook_cd(testloader, device)     # Cell 18
        PI_proj_tensor = make_PI_proj_cd(PI_matrices, num_models=40)  # Cell 19
        out_path = IT_DIR / "interaction_tensor_B_SCD_model.pt"

    # Step 3 — correlation + threshold (Cell 32-33 / Cell 20)
    corr_tensor = compute_corr(PI_proj_tensor)
    print(corr_tensor.shape)
    gamma_corr = get_gamma_corr(corr_tensor, args.corr_percentile)

    # Step 4 — clustering (Cell 34 / Cell 21)
    assignment = cluster_features(corr_tensor, gamma_corr)
    print(assignment.shape)

    # Step 5 — binary mask (Cell 35 / Cell 22)
    data_feature = compute_data_feature(PI_proj_tensor, args.data_threshold)
    print(data_feature.shape)

    # Step 6 — assemble Ω (Cell 36 / Cell 23)
    interaction_tensor = compute_interaction_tensor(assignment, data_feature)
    torch.save(interaction_tensor, out_path)
    print(interaction_tensor.shape)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
