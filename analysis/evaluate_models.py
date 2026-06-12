"""
Table 2: Accuracy evaluation of all methods × architectures.

Loads checkpoints from the checkpoints/ directory and evaluates each
configuration (3 runs each), printing Top-1 and Top-5 mean ± std.

Architectures: ResNet-18, ResNet-34, ResNet-50, DenseNet-121
Methods:       Baseline, CS-KD, PS-KD, CD-200ep, CD-300ep
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
          f"Top-5: {mean5:.2f} ± {std5:.2f}%   runs={accs1}")
    return mean1, std1


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods for Table 2")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    testloader = make_testloader(args.data_dir)

    # ── ResNet-18 ──────────────────────────────────────────────────────────
    print("\n═══════════════ ResNet-18 ═══════════════")

    print("\n[Baseline]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet18",
        [
            # fill in your 3 checkpoint paths, e.g.:
            # f"{CKPT_DIR}/resnet/resnet18/<run1>/resnet18-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CS-KD]")
    build_and_eval(
        f"{BASE_DIR}/cs-kd/models/resnet.py", "CIFAR_ResNet18",
        [
            # f"{CKPT_DIR}/cs_kd/results/cifar100/CIFAR_ResNet18/<run>/ckpt.t7",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[PS-KD]")
    build_and_eval(
        f"{BASE_DIR}/ps-kd/models/preact_resnet.py", "CIFAR_ResNet18_preActBasic",
        [
            # f"{CKPT_DIR}/ps_kd/runs/pskd_resnet18_cifar100/<run>/model/checkpoint_best.pth",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[CD (200 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet18",
        [
            # f"{CKPT_DIR}/resnet/resnet18/<run>/resnet18-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CD (300 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet18",
        [
            # f"{CKPT_DIR}/resnet/resnet18/<run>/resnet18-XXX-best.pth",
        ],
        testloader, device,
    )

    # ── ResNet-34 ──────────────────────────────────────────────────────────
    print("\n═══════════════ ResNet-34 ═══════════════")

    print("\n[Baseline]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet34",
        [
            # f"{CKPT_DIR}/resnet/resnet34/<run>/resnet34-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CS-KD]")
    build_and_eval(
        f"{BASE_DIR}/cs-kd/models/resnet.py", "CIFAR_ResNet34",
        [
            # f"{CKPT_DIR}/cs_kd/results/cifar100/CIFAR_ResNet34/<run>/ckpt.t7",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[PS-KD]")
    build_and_eval(
        f"{BASE_DIR}/ps-kd/models/preact_resnet.py", "CIFAR_ResNet34_preActBasic",
        [
            # f"{CKPT_DIR}/ps_kd/runs/pskd_resnet34_cifar100/<run>/model/checkpoint_best.pth",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[CD (200 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet34",
        [
            # f"{CKPT_DIR}/resnet/resnet34/<run>/resnet34-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CD (300 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet34",
        [
            # f"{CKPT_DIR}/resnet/resnet34/<run>/resnet34-XXX-best.pth",
        ],
        testloader, device,
    )

    # ── ResNet-50 ──────────────────────────────────────────────────────────
    print("\n═══════════════ ResNet-50 ═══════════════")

    print("\n[Baseline]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet50",
        [
            # f"{CKPT_DIR}/resnet/resnet50/<run>/resnet50-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CS-KD]")
    build_and_eval(
        f"{BASE_DIR}/cs-kd/models/resnet.py", "CIFAR_ResNet50",
        [
            # f"{CKPT_DIR}/cs_kd/results/cifar100/CIFAR_ResNet50/<run>/ckpt.t7",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[PS-KD]")
    build_and_eval(
        f"{BASE_DIR}/ps-kd/models/preact_resnet.py", "CIFAR_ResNet50_Bottle",
        [
            # f"{CKPT_DIR}/ps_kd/runs/pskd_resnet50_cifar100/<run>/model/checkpoint_best.pth",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[CD (200 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet50",
        [
            # f"{CKPT_DIR}/resnet/resnet50/<run>/resnet50-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CD (300 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/resnet.py", "resnet50",
        [
            # f"{CKPT_DIR}/resnet/resnet50/<run>/resnet50-XXX-best.pth",
        ],
        testloader, device,
    )

    # ── DenseNet-121 ───────────────────────────────────────────────────────
    print("\n═══════════════ DenseNet-121 ═══════════════")

    print("\n[Baseline]")
    build_and_eval(
        f"{TRAIN_DIR}/models/densenet.py", "densenet121",
        [
            # f"{CKPT_DIR}/densenet121/<run>/densenet121-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CS-KD]")
    build_and_eval(
        f"{BASE_DIR}/cs-kd/models/densenet3.py", "CIFAR_DenseNet121",
        [
            # f"{CKPT_DIR}/cs_kd/results/cifar100/CIFAR_DenseNet121/<run>/ckpt.t7",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[PS-KD]")
    build_and_eval(
        f"{BASE_DIR}/ps-kd/models/densenet_cifar.py", "CIFAR_DenseNet121",
        [
            # f"{CKPT_DIR}/ps_kd/runs/pskd_densenet121_cifar100/<run>/model/checkpoint_best.pth",
        ],
        testloader, device, builder_kwargs={"num_classes": 100},
    )

    print("\n[CD (200 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/densenet.py", "densenet121",
        [
            # f"{CKPT_DIR}/densenet121/<run>/densenet121-XXX-best.pth",
        ],
        testloader, device,
    )

    print("\n[CD (300 ep)]")
    build_and_eval(
        f"{TRAIN_DIR}/models/densenet.py", "densenet121",
        [
            # f"{CKPT_DIR}/densenet121/<run>/densenet121-XXX-best.pth",
        ],
        testloader, device,
    )


if __name__ == "__main__":
    main()