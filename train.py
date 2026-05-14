#!/usr/bin/env python3
"""
Training script for UNet mammogram segmentation on CBIS-DDSM.

Usage:
    python train.py \
        --train_csv /data/mass_case_description_train_set.csv \
        --test_csv  /data/mass_case_description_test_set.csv \
        --data_root /data/CBIS-DDSM \
        --out_dir   ./checkpoints
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from unet import UNet
from datasets import CBISDDSMSegDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """BCE + soft Dice — standard combo for binary medical segmentation."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)

        # Soft Dice
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + smooth) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
        )
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    smooth = 1e-5
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    score = (2 * intersection + smooth) / (
        pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    )
    return score.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    smooth = 1e-5
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = total_dice = total_iou = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv",  required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir",   default="checkpoints")
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--init_features", type=int, default=32)
    p.add_argument("--bce_weight", type=float, default=0.5,
                   help="Weight for BCE in the BCE+Dice combo loss (0-1)")
    p.add_argument("--patience",  type=int, default=10,
                   help="Early-stopping patience (epochs without val improvement)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    image_size = tuple(args.image_size)

    train_ds = CBISDDSMSegDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        image_size=image_size,
        augment=True,
    )
    val_ds = CBISDDSMSegDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Model — dataset yields (1, H, W) grayscale; UNet defaults to 3
    # ------------------------------------------------------------------
    model = UNet(in_channels=1, out_channels=1, init_features=args.init_features)
    model = model.to(device)

    criterion = DiceBCELoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_dice, val_iou = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        scheduler.step(val_dice)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} iou={train_iou:.4f}  |  "
            f"val loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val_dice,
            "args": vars(args),
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  -> New best val dice: {best_val_dice:.4f}  (saved best.pt)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    print(f"\nTraining complete. Best val dice: {best_val_dice:.4f}")
    print(f"Checkpoints saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
