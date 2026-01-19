# estimate_training_cost.py
# ============================================================================
# Estimate Training Compute Cost (WITHOUT retraining)
# ============================================================================
# What it does:
#   - For each model config in configs.py, builds the model + criterion + optimizer
#   - Runs a short timing profile over N batches:
#       * training step timing: forward + backward + optimizer step
#       * validation timing (optional): forward only
#   - Extrapolates to full training cost: epochs × steps_per_epoch
#   - Writes JSON + CSV summaries
#
# Usage examples:
#   python estimate_training_cost.py --profile-batches 50 --warmup-batches 5 --epochs 30
#   python estimate_training_cost.py --profile-batches 80 --include-val --epochs 50
#
# Notes:
#   - This estimates *wall-clock* training time on your current machine/GPU.
#   - It does NOT save models/checkpoints.
#   - Uses CUDA synchronization for accurate GPU timing.
# ============================================================================

import os
import json
import math
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from configs import configs, PROCESSED_PATH, WAVELETS_PATH, RESULTS_PATH
from models import (
    CWT2DCNN, DualStreamCNN, ViTFusionECG,
    SwinTransformerECG, SwinTransformerEarlyFusion,
    ViTLateFusion, EfficientNetLateFusion,
    SwinTransformerLateFusion, HybridSwinTransformerECG,
    HybridSwinTransformerEarlyFusion, HybridSwinTransformerLateFusion,
    EfficientNetFusionECG, EfficientNetEarlyFusion, EfficientNetLateFusion,
    ResNet50EarlyFusion, ResNet50LateFusion, ResNet50ECG, EfficientNetECG
)
from focal_loss import FocalLoss, DistributionAwareFocalLoss


# ----------------------------
# Defaults (match your training)
# ----------------------------
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset
# ============================================================================
class CWTDataset(Dataset):
    """Memory-efficient dataset for CWT data via memmap."""

    def __init__(self, scalo_path, phaso_path, labels, mode="scalogram", augment=False):
        self.scalograms = np.load(scalo_path, mmap_mode="r")
        self.phasograms = np.load(phaso_path, mmap_mode="r")
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
        phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
        y = self.labels[idx]

        if self.mode == "scalogram":
            return scalo, y
        elif self.mode == "phasogram":
            return phaso, y
        elif self.mode == "both":
            return (scalo, phaso), y
        elif self.mode == "fusion":
            fused = torch.cat([scalo, phaso], dim=0)  # (24, H, W)
            return fused, y
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================================
# Helpers
# ============================================================================
def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_lr_from_model_name(model_name: str, default_lr: float = 1e-3) -> float:
    """Mirror the LR logic you used in training."""
    if "Swin" in model_name or "HybridSwin" in model_name:
        return 3e-5
    if "ViT" in model_name:
        return 5e-5
    if "EfficientNet" in model_name:
        return 1e-4
    if "Enhanced" in model_name or "XResNet" in model_name:
        return 1e-3
    return default_lr


def build_model_from_config(cfg: dict, num_classes: int) -> nn.Module:
    mode = cfg.get("mode", "scalogram")
    adapter_strategy = cfg.get("adapter", "learned")

    model_key = cfg["model"]

    if model_key == "DualStream":
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif model_key == "CWT2DCNN":
        num_ch = 24 if mode == "fusion" else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif model_key == "ViTFusionECG":
        model = ViTFusionECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "SwinTransformerECG":
        model = SwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "SwinTransformerEarlyFusion":
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif model_key == "SwinTransformerLateFusion":
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "ViTLateFusion":
        model = ViTLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "HybridSwinTransformerECG":
        model = HybridSwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "HybridSwinTransformerEarlyFusion":
        model = HybridSwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif model_key == "HybridSwinTransformerLateFusion":
        model = HybridSwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "EfficientNetFusionECG":
        model = EfficientNetFusionECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "EfficientNetECG":
        model = EfficientNetECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "EfficientNetEarlyFusion":
        model = EfficientNetEarlyFusion(num_classes=num_classes, pretrained=True)
    elif model_key == "EfficientNetLateFusion":
        model = EfficientNetLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "ResNet50EarlyFusion":
        model = ResNet50EarlyFusion(num_classes=num_classes, pretrained=True)
    elif model_key == "ResNet50LateFusion":
        model = ResNet50LateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_key == "ResNet50ECG":
        model = ResNet50ECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    else:
        raise ValueError(f"Unknown model in config: {model_key}")

    return model.to(DEVICE)


def build_criterion(cfg: dict, metadata: dict) -> nn.Module:
    """
    Mirrors your training choices.
    If you want to estimate cost across different loss modes, this must match the real training.
    """
    loss_type = cfg.get("loss", "bce")

    if loss_type == "focal":
        return FocalLoss()

    if loss_type == "focal_weighted":
        # Use the same style as your training script (distribution-aware focal).
        y_train = np.load(os.path.join(PROCESSED_PATH, "y_train.npy"))
        class_counts = y_train.sum(axis=0).astype(np.float32)
        total_samples = float(len(y_train))
        # Avoid divide-by-zero:
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = torch.FloatTensor(total_samples / (len(metadata["classes"]) * class_counts))
        return DistributionAwareFocalLoss(
            class_weights=class_weights,
            gamma=2.0,
            alpha=0.25,
            reduction="mean"
        )

    if loss_type == "focal_adaptive":
        return FocalLoss()

    return nn.BCEWithLogitsLoss()


def is_dual_mode(cfg: dict) -> bool:
    mode = cfg.get("mode", "")
    return (cfg.get("model") == "DualStream") or (mode == "both")


@torch.no_grad()
def _forward_only(model: nn.Module, batch, cfg: dict):
    """Forward pass only (validation-like)."""
    if isinstance(batch[0], (tuple, list)):
        (x1, x2), _y = batch
        x1 = x1.to(DEVICE, non_blocking=True)
        x2 = x2.to(DEVICE, non_blocking=True)

        if is_dual_mode(cfg):
            _ = model(x1, x2)
        else:
            mode = cfg.get("mode")
            if mode == "scalogram":
                _ = model(x1)
            elif mode == "phasogram":
                _ = model(x2)
            elif mode == "fusion":
                _ = model(torch.cat([x1, x2], dim=1))
            else:
                _ = model(x1)
    else:
        x, _y = batch
        x = x.to(DEVICE, non_blocking=True)
        _ = model(x)


def _train_step(model: nn.Module, batch, cfg: dict, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    """One training step: forward + loss + backward + optimizer step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if isinstance(batch[0], (tuple, list)):
        (x1, x2), y = batch
        x1 = x1.to(DEVICE, non_blocking=True)
        x2 = x2.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if is_dual_mode(cfg):
            out = model(x1, x2)
        else:
            mode = cfg.get("mode")
            if mode == "scalogram":
                out = model(x1)
            elif mode == "phasogram":
                out = model(x2)
            elif mode == "fusion":
                out = model(torch.cat([x1, x2], dim=1))
            else:
                out = model(x1)
    else:
        x, y = batch
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x)

    loss = criterion(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def profile_training_time(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    profile_batches: int,
    warmup_batches: int
) -> dict:
    """Profiles average training step time over a limited number of batches."""
    model.train()

    # Warmup
    it = iter(train_loader)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        _train_step(model, batch, cfg, criterion, optimizer)
    _sync_cuda()

    # Timed
    times = []
    losses = []
    it = iter(train_loader)
    for _ in range(profile_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        t0 = time.perf_counter()
        loss = _train_step(model, batch, cfg, criterion, optimizer)
        _sync_cuda()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        losses.append(loss)

    if not times:
        return {"avg_step_s": None, "std_step_s": None, "avg_loss": None, "steps_measured": 0}

    return {
        "avg_step_s": float(np.mean(times)),
        "std_step_s": float(np.std(times)),
        "avg_loss": float(np.mean(losses)),
        "steps_measured": int(len(times)),
    }


@torch.no_grad()
def profile_validation_time(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: dict,
    profile_batches: int,
    warmup_batches: int
) -> dict:
    """Profiles average forward-only time over a limited number of batches."""
    model.eval()

    # Warmup
    it = iter(val_loader)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        _forward_only(model, batch, cfg)
    _sync_cuda()

    # Timed
    times = []
    it = iter(val_loader)
    for _ in range(profile_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        t0 = time.perf_counter()
        _forward_only(model, batch, cfg)
        _sync_cuda()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if not times:
        return {"avg_forward_s": None, "std_forward_s": None, "steps_measured": 0}

    return {
        "avg_forward_s": float(np.mean(times)),
        "std_forward_s": float(np.std(times)),
        "steps_measured": int(len(times)),
    }


def estimate_total_time_seconds(
    train_size: int,
    val_size: int,
    batch_size: int,
    epochs: int,
    avg_train_step_s: float,
    avg_val_step_s: float | None,
    validate_each_epoch: bool
) -> dict:
    steps_per_epoch = int(math.ceil(train_size / batch_size))
    val_steps = int(math.ceil(val_size / batch_size))

    train_total = steps_per_epoch * epochs * avg_train_step_s

    val_total = 0.0
    if validate_each_epoch and (avg_val_step_s is not None):
        val_total = val_steps * epochs * avg_val_step_s

    total = train_total + val_total

    return {
        "steps_per_epoch": steps_per_epoch,
        "val_steps_per_epoch": val_steps,
        "train_total_s": float(train_total),
        "val_total_s": float(val_total),
        "total_s": float(total),
        "total_gpu_hours": float(total / 3600.0),
    }


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in fieldnames) + "\n")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs to estimate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--profile-batches", type=int, default=50, help="Batches to time per model.")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches (not timed).")
    parser.add_argument("--include-val", action="store_true", help="Also estimate validation cost per epoch.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save JSON/CSV outputs.")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(RESULTS_PATH, "compute_cost_estimates")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ESTIMATE TRAINING COMPUTE COST (NO FULL RETRAINING)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Profile batches: {args.profile_batches} | Warmup: {args.warmup_batches}")
    print(f"Epochs estimate: {args.epochs} | Batch size: {args.batch_size} | Workers: {args.num_workers}")
    print(f"Include validation: {args.include_val}")
    print(f"Output dir: {out_dir}")

    # Load metadata
    with open(os.path.join(PROCESSED_PATH, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    train_size = int(metadata["train_size"])
    val_size = int(metadata["val_size"])
    num_classes = int(metadata["num_classes"])

    # Load labels (just once)
    y_train = np.load(os.path.join(PROCESSED_PATH, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_PATH, "y_val.npy"))

    # Common paths
    train_scalo = os.path.join(WAVELETS_PATH, "train_scalograms.npy")
    train_phaso = os.path.join(WAVELETS_PATH, "train_phasograms.npy")
    val_scalo = os.path.join(WAVELETS_PATH, "val_scalograms.npy")
    val_phaso = os.path.join(WAVELETS_PATH, "val_phasograms.npy")

    results = {}
    csv_rows = []

    for cfg in configs:
        name = cfg.get("name", f"{cfg.get('model')}_{cfg.get('mode')}")
        mode = cfg.get("mode", "scalogram")

        print("\n" + "-" * 80)
        print(f"Profiling: {name}")
        print(f"  model={cfg.get('model')} | mode={mode} | loss={cfg.get('loss','bce')}")
        print("-" * 80)

        # Build datasets/loaders for this mode
        train_ds = CWTDataset(train_scalo, train_phaso, y_train, mode=mode, augment=True)
        val_ds = CWTDataset(val_scalo, val_phaso, y_val, mode=mode, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Build model/criterion/optimizer
        model = build_model_from_config(cfg, num_classes=num_classes)
        criterion = build_criterion(cfg, metadata)
        lr = infer_lr_from_model_name(cfg.get("model", ""), default_lr=1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Profile training time per step
        train_profile = profile_training_time(
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            criterion=criterion,
            optimizer=optimizer,
            profile_batches=args.profile_batches,
            warmup_batches=args.warmup_batches
        )

        # Profile validation forward-only time per step (optional)
        val_profile = None
        if args.include_val:
            val_profile = profile_validation_time(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                profile_batches=min(args.profile_batches, 50),
                warmup_batches=min(args.warmup_batches, 5)
            )

        # Extrapolate totals
        avg_train_step_s = train_profile["avg_step_s"]
        if avg_train_step_s is None:
            print("  [Skip] Could not measure training time (no batches?)")
            continue

        avg_val_step_s = None
        if args.include_val and val_profile is not None:
            avg_val_step_s = val_profile["avg_forward_s"]

        totals = estimate_total_time_seconds(
            train_size=train_size,
            val_size=val_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            avg_train_step_s=avg_train_step_s,
            avg_val_step_s=avg_val_step_s,
            validate_each_epoch=args.include_val
        )

        # Pack results
        results[name] = {
            "config": cfg,
            "device": str(DEVICE),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs_estimated": args.epochs,
            "train_profile": train_profile,
            "val_profile": val_profile,
            "totals": totals
        }

        # Print quick summary
        print(f"  Avg train step: {train_profile['avg_step_s']:.4f}s (std {train_profile['std_step_s']:.4f}s) "
              f"| measured {train_profile['steps_measured']} steps")
        if args.include_val and (val_profile is not None) and (val_profile["avg_forward_s"] is not None):
            print(f"  Avg val step:   {val_profile['avg_forward_s']:.4f}s (std {val_profile['std_forward_s']:.4f}s) "
                  f"| measured {val_profile['steps_measured']} steps")

        print(f"  Estimated total: {totals['total_s'] / 60.0:.2f} min "
              f"({totals['total_gpu_hours']:.3f} GPU-hrs) "
              f"| train {totals['train_total_s'] / 60.0:.2f} min "
              f"+ val {totals['val_total_s'] / 60.0:.2f} min")

        csv_rows.append({
            "model_name": name,
            "model": cfg.get("model"),
            "mode": cfg.get("mode"),
            "loss": cfg.get("loss", "bce"),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs_estimated": args.epochs,
            "avg_train_step_s": train_profile["avg_step_s"],
            "std_train_step_s": train_profile["std_step_s"],
            "avg_val_step_s": (val_profile["avg_forward_s"] if (val_profile is not None) else ""),
            "steps_per_epoch": totals["steps_per_epoch"],
            "val_steps_per_epoch": totals["val_steps_per_epoch"],
            "train_total_s": totals["train_total_s"],
            "val_total_s": totals["val_total_s"],
            "total_s": totals["total_s"],
            "total_gpu_hours": totals["total_gpu_hours"],
            "device": str(DEVICE),
        })

        # Free VRAM between models
        del model
        torch.cuda.empty_cache()

    # Save outputs
    json_path = os.path.join(out_dir, "training_compute_cost_estimates.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    csv_path = os.path.join(out_dir, "training_compute_cost_estimates.csv")
    fieldnames = [
        "model_name", "model", "mode", "loss",
        "batch_size", "num_workers", "epochs_estimated",
        "avg_train_step_s", "std_train_step_s", "avg_val_step_s",
        "steps_per_epoch", "val_steps_per_epoch",
        "train_total_s", "val_total_s", "total_s", "total_gpu_hours",
        "device"
    ]
    write_csv(csv_path, csv_rows, fieldnames)
    print(f"Saved CSV:  {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()