"""
Table 2: Accuracy evaluation of all methods × architectures.

Loads checkpoints from the checkpoints/ directory and evaluates each
configuration (3 runs each), printing Top-1 and Top-5 mean ± std.

Architectures: ResNet-18, ResNet-34, ResNet-50, DenseNet-121
Methods:       Baseline, CS-KD, PS-KD, CD-200ep, CD-300ep

Checkpoints are hosted on HuggingFace (KangWooya/WDSL-tensors, under
checkpoints/). Download them into ./checkpoints/ (see README) and run:

    uv run python analysis/evaluate_models.py

The checkpoints/ layout mirrors the paths below:
    checkpoints/table2/<method>/<arch>/run{1,2,3}.pth
"""

import argparse
import importlib.util
import os
import re
import statistics
import types
from pathlib import Path
from typing import Any, Dict

import torch
import torchvision
import torchvision.transforms as transforms

# ── Path Configuration ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = Path("data/")   # set to your CIFAR-100 root
CKPT_DIR  = str(REPO_ROOT / "checkpoints")
TRAIN_DIR = str(REPO_ROOT / "training")
BASE_DIR  = str(REPO_ROOT / "baselines")
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


# ── Dynamic model loading ──────────────────────────────────────────────────
def import_from_path(module_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(module: types.ModuleType, builder_name: str, **builder_kwargs) -> torch.nn.Module:
    if not hasattr(module, builder_name):
        cand = [n for n in dir(module) if n.lower() == builder_name.lower()]
        if not cand:
            raise AttributeError(
                f"Builder '{builder_name}' not found. Available: {sorted(dir(module))[:50]}"
            )
        builder_name = cand[0]
    builder = getattr(module, builder_name)
    model = builder(**builder_kwargs)
    if isinstance(model, type):
        model = model(**builder_kwargs)
    return model


def _strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    return {re.sub(r'^module\.', '', k): v for k, v in sd.items()}


def _best_state_dict_key(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    for key in ['state_dict', 'net', 'model', 'ema_state_dict']:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("state_dict not found in checkpoint")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str,
                    map_location='cpu', strict: bool = False) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, (list, tuple)):
        ckpt = ckpt[0]
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    sd = _strip_module_prefix(_best_state_dict_key(ckpt))
    try:
        model.load_state_dict(sd, strict=strict)
    except Exception as e:
        print(f"[warn] strict load failed → non-strict. Reason: {e}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:     print("[warn] Missing keys:", missing[:8])
        if unexpected:  print("[warn] Unexpected keys:", unexpected[:8])
    return model
# ──────────────────────────────────────────────────────────────────────────


def test(model, test_loader, device):
    model.to(device).eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(images).topk(5, dim=1)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            top1 += correct[:, :1].sum().item()
            top5 += correct.sum().item()
            total += labels.size(0)
    return 100 * top1 / total, 100 * top5 / total


def build_and_eval(model_py, builder_name, ckpt_paths, testloader, device, builder_kwargs=None):
    builder_kwargs = builder_kwargs or {}
    if not ckpt_paths:
        print(f"  [{builder_name}]  (skipped — no checkpoints provided)")
        return None, None
    missing = [p for p in ckpt_paths if not os.path.exists(p)]
    if missing:
        print(f"  [{builder_name}]  (skipped — {len(missing)}/{len(ckpt_paths)} checkpoints not found; "
              f"download from HuggingFace, see README)")
        return None, None
    m = import_from_path(f"m_{hash(model_py) % 10**8}", model_py)
    accs1, accs5 = [], []
    for p in ckpt_paths:
        model = build_model(m, builder_name, **builder_kwargs)
        model = load_checkpoint(model, p, map_location='cpu', strict=False)
        a1, a5 = test(model, testloader, device)
        accs1.append(a1); accs5.append(a5)
    mean1, std1 = statistics.mean(accs1), statistics.pstdev(accs1)
    mean5, std5 = statistics.mean(accs5), statistics.pstdev(accs5)
    print(f"  [{builder_name}]  Top-1: {mean1:.2f} ± {std1:.2f}%   "
          f"Top-5: {mean5:.2f} ± {std5:.2f}%   runs={[round(a, 2) for a in accs1]}")
    return mean1, std1


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods for Table 2")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--ckpt-dir", default=CKPT_DIR,
                        help="Root of the downloaded checkpoints/ directory")
    args = parser.parse_args()

    ckpt = args.ckpt_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    testloader = make_testloader(args.data_dir)

    # Model builders per method (CIFAR-100 variants from each source repo)
    RESNET = f"{TRAIN_DIR}/models/resnet.py"          # baseline + CD (pytorch-cifar100)
    DENSE  = f"{TRAIN_DIR}/models/densenet.py"
    CSKD_R = f"{BASE_DIR}/cs-kd/models/resnet.py"
    CSKD_D = f"{BASE_DIR}/cs-kd/models/densenet3.py"
    PSKD_R = f"{BASE_DIR}/ps-kd/models/preact_resnet.py"
    PSKD_D = f"{BASE_DIR}/ps-kd/models/densenet_cifar.py"

    def runs(method, arch):
        return [f"{ckpt}/table2/{method}/{arch}/run{i}.pth" for i in (1, 2, 3)]

    # (label, model_py, builder, ckpt_list, kwargs)
    plan = {
        "resnet18": [
            ("Baseline",    RESNET, "resnet18",                    runs("baseline", "resnet18"), None),
            ("CS-KD",       CSKD_R, "CIFAR_ResNet18",              runs("cs-kd", "resnet18"), {"num_classes": 100}),
            ("PS-KD",       PSKD_R, "CIFAR_ResNet18_preActBasic",  runs("ps-kd", "resnet18"), {"num_classes": 100}),
            ("CD (200 ep)", RESNET, "resnet18",                    runs("cd-200", "resnet18"), None),
            ("CD (300 ep)", RESNET, "resnet18",                    runs("cd-300", "resnet18"), None),
        ],
        "resnet34": [
            ("Baseline",    RESNET, "resnet34",                    runs("baseline", "resnet34"), None),
            ("CS-KD",       CSKD_R, "CIFAR_ResNet34",              runs("cs-kd", "resnet34"), {"num_classes": 100}),
            ("PS-KD",       PSKD_R, "CIFAR_ResNet34_preActBasic",  runs("ps-kd", "resnet34"), {"num_classes": 100}),
            ("CD (200 ep)", RESNET, "resnet34",                    runs("cd-200", "resnet34"), None),
            ("CD (300 ep)", RESNET, "resnet34",                    runs("cd-300", "resnet34"), None),
        ],
        "resnet50": [
            ("Baseline",    RESNET, "resnet50",                    runs("baseline", "resnet50"), None),
            ("CS-KD",       CSKD_R, "CIFAR_ResNet50",              runs("cs-kd", "resnet50"), {"num_classes": 100}),
            ("PS-KD",       PSKD_R, "CIFAR_ResNet50_Bottle",       runs("ps-kd", "resnet50"), {"num_classes": 100}),
            ("CD (200 ep)", RESNET, "resnet50",                    runs("cd-200", "resnet50"), None),
            ("CD (300 ep)", RESNET, "resnet50",                    runs("cd-300", "resnet50"), None),
        ],
        "densenet121": [
            ("Baseline",    DENSE,  "densenet121",                 runs("baseline", "densenet121"), None),
            ("CS-KD",       CSKD_D, "CIFAR_DenseNet121",           runs("cs-kd", "densenet121"), {"num_classes": 100}),
            ("PS-KD",       PSKD_D, "CIFAR_DenseNet121",           runs("ps-kd", "densenet121"), {"num_classes": 100}),
            # DenseNet-121 CD-200ep: original checkpoints were lost; these are freshly
            # retrained with the same recipe (reproduce at 79.15 ± 0.22; paper 78.71 ± 0.18).
            ("CD (200 ep)", DENSE,  "densenet121",                 runs("cd-200", "densenet121"), None),
            ("CD (300 ep)", DENSE,  "densenet121",                 runs("cd-300", "densenet121"), None),
        ],
    }

    for arch, rows in plan.items():
        print(f"\n═══════════════ {arch} ═══════════════")
        for label, model_py, builder, ckpts, kwargs in rows:
            print(f"\n[{label}]")
            build_and_eval(model_py, builder, ckpts, testloader, device, builder_kwargs=kwargs)


if __name__ == "__main__":
    main()
