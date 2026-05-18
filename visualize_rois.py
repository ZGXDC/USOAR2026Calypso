#!/usr/bin/env python3
"""
Visualize ROIs from CBIS-DDSM-style CSV by overlaying mask contours
on the full mammogram image.

Usage:
    python visualize_rois.py metadata.csv --data-root /path/to/dicom/root
    python visualize_rois.py metadata.csv --data-root /path/to/dicom/root --row 0
    python visualize_rois.py metadata.csv --data-root /path/to/dicom/root --patient P_00001
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def resolve_dicom(data_root: Path, csv_path: str) -> Path | None:
    """
    CSV paths look like: .../SeriesUID/000000.dcm
    But NBIA-downloaded files are renamed to UUIDs like e35e53fd-....dcm.
    Strip the filename, then return the first .dcm file in the directory.
    """
    csv_path = csv_path.strip()
    full = data_root / csv_path

    # First try the literal path (works for Kaggle mirrors, etc.)
    if full.exists():
        return full

    # Fall back: use the directory and find whatever .dcm lives there
    directory = full.parent
    if not directory.is_dir():
        return None

    dicoms = sorted(directory.glob("*.dcm"))
    if not dicoms:
        return None

    # If the CSV filename ends in a number (e.g. 000001.dcm for the mask),
    # try to pick that index when multiple DICOMs exist in one folder.
    stem = full.stem  # "000000" or "000001"
    if stem.isdigit() and len(dicoms) > 1:
        idx = int(stem)
        if idx < len(dicoms):
            return dicoms[idx]

    return dicoms[0]


def load_dicom(path: Path) -> np.ndarray:
    """Read a DICOM file and return its pixel array as float."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    # Normalize to [0, 1] for display
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def bbox_from_mask(mask: np.ndarray):
    """Return (rmin, rmax, cmin, cmax) of nonzero region, or None."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def visualize_row(row: pd.Series, data_root: Path):
    img_path  = resolve_dicom(data_root, row["image file path"])
    mask_path = resolve_dicom(data_root, row["ROI mask file path"])
    crop_path = resolve_dicom(data_root, row["cropped image file path"])

    if img_path is None:
        print(f"[skip] missing image for {row['patient_id']}")
        return
    if mask_path is None:
        print(f"[skip] missing mask for {row['patient_id']}")
        return

    image = load_dicom(img_path)
    mask = load_dicom(mask_path)
    mask_bin = mask > 0.5

    # Swap heuristic: CBIS-DDSM sometimes has mask/cropped paths reversed
    if mask.shape != image.shape or np.unique((mask * 255).astype(np.uint8)).size > 10:
        if crop_path is not None:
            alt = load_dicom(crop_path)
            if alt.shape == image.shape:
                mask = alt
                mask_bin = mask > 0.5

    title = (
        f"{row['patient_id']} {row['left or right breast']} {row['image view']} "
        f"| {row['pathology']} | {row.get('mass shape', '')}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Mammogram")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~mask_bin, mask_bin), cmap="autumn", alpha=0.4)
    bb = bbox_from_mask(mask_bin)
    if bb is not None:
        rmin, rmax, cmin, cmax = bb
        axes[1].add_patch(
            Rectangle(
                (cmin, rmin), cmax - cmin, rmax - rmin,
                fill=False, edgecolor="lime", linewidth=2,
            )
        )
    axes[1].set_title("Mammogram + ROI overlay")
    axes[1].axis("off")

    # Cropped zoom around the ROI
    if bb is not None:
        rmin, rmax, cmin, cmax = bb
        pad = 50
        r0, r1 = max(0, rmin - pad), min(image.shape[0], rmax + pad)
        c0, c1 = max(0, cmin - pad), min(image.shape[1], cmax + pad)
        axes[2].imshow(image[r0:r1, c0:c1], cmap="gray")
        axes[2].set_title("ROI zoom")
    else:
        axes[2].set_title("ROI zoom (no mask)")
    axes[2].axis("off")

    plt.tight_layout()
    # plt.show()
    #SAVING FIGURES TO "SavedFigures" DIRECTORY TO MAKE IMAGES ACCESSIBLE
    savedPath = title.replace(" ", "_")
    savedPath = savedPath.replace("|", "")
    savedPath = savedPath.replace("/", "_")
    savedPath = f"SavedFigures/{savedPath}.png"
    plt.savefig(savedPath, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Metadata CSV file")
    p.add_argument("--data-root", default=".", help="Root directory containing the DICOM folders")
    p.add_argument("--row", type=int, help="Show a specific row index")
    p.add_argument("--patient", help="Filter by patient_id, e.g. P_00001")
    p.add_argument("--limit", type=int, default=5, help="Max rows to display")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    data_root = Path(args.data_root)

    if args.row is not None:
        visualize_row(df.iloc[args.row], data_root)
        return

    if args.patient:
        df = df[df["patient_id"] == args.patient]

    for _, row in df.head(args.limit).iterrows():
        visualize_row(row, data_root)


if __name__ == "__main__":
    main()