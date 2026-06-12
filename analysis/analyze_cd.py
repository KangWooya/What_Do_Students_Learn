"""
Figure 3: Feature-level analysis of Confusion Distillation via Interaction Tensor.

Loads pre-computed interaction_tensor_B_SCD_model.pt (40 models):
  [0:20]  = Baseline (ResNet-18)
  [20:40] = CD model (ResNet-18)

Produces:
  Figure 3a — Feature frequency distribution (Baseline vs. CD)
  Figure 3b — Feature count per sample (Baseline vs. CD)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# ── Path Configuration ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
IT_DIR    = str(REPO_ROOT / "interaction_tensors")
# ──────────────────────────────────────────────────────────────────────────


# ── Figure 3a: feature frequency distribution ─────────────────────────────
def plot_feature_frequency_all(omega, out_path=None):
    model_slices = {
        "Bmodel":  omega[0:20],
        "CDmodel": omega[20:40],
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


# ── Figure 3b: feature count per sample ───────────────────────────────────
def plot_data_feature_count_all(omega, out_path=None):
    model_slices = {
        "Bmodel":  omega[0:20],
        "CDmodel": omega[20:40],
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


def main():
    parser = argparse.ArgumentParser(description="CD feature analysis (Figure 3)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to save figures (default: display only)")
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.out_dir, name) if args.out_dir else None

    print("Loading interaction tensor …")
    it_path = os.path.join(IT_DIR, "interaction_tensor_B_SCD_model.pt")
    interaction_tensor = torch.load(it_path, weights_only=True)
    print(f"  shape: {interaction_tensor.shape}")  # (40, N, T)

    print("\n[Figure 3a] Feature frequency distribution …")
    plot_feature_frequency_all(interaction_tensor, out("fig3a_feature_frequency.png"))

    print("[Figure 3b] Feature count per sample …")
    plot_data_feature_count_all(interaction_tensor, out("fig3b_data_feature_count.png"))


if __name__ == "__main__":
    main()