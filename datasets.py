#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader for CBIS-DDSM segmentation (U-Net training).

Each sample yields:
    image: float32 tensor, shape (1, H, W), normalized to [0, 1]
    mask:  float32 tensor, shape (1, H, W), binary {0, 1}
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


# --------------------------------------------------------------------------
# Path + DICOM helpers (same logic as the visualization script)
# --------------------------------------------------------------------------

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
    """Read DICOM, return float32 array normalized to [0, 1]."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def pick_mask(mask_arr: np.ndarray, crop_arr: Optional[np.ndarray],
              image_shape: tuple) -> np.ndarray:
    """
    CBIS-DDSM sometimes swaps the 'ROI mask' and 'cropped image' paths.
    The real mask is binary (2 unique values) and matches the full image shape.
    """
    def looks_like_mask(a):
        if a is None or a.shape != image_shape:
            return False
        # Binary-ish: very few unique values when quantized
        return np.unique((a * 255).astype(np.uint8)).size <= 10

    if looks_like_mask(mask_arr):
        return mask_arr
    if looks_like_mask(crop_arr):
        return crop_arr
    # Fallback: whichever matches the image shape
    if mask_arr.shape == image_shape:
        return mask_arr
    if crop_arr is not None and crop_arr.shape == image_shape:
        return crop_arr
    raise ValueError("Neither mask nor crop matches the image shape.")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class CBISDDSMSegDataset(Dataset):
    """
    CBIS-DDSM segmentation dataset for U-Net training.

    Args:
        csv_path:   path to mass_case_description_{train,test}_set.csv
        data_root:  directory containing Mass-Training_P_* / Mass-Test_P_* folders
        image_size: output (H, W) after resizing. Mammograms are huge (~3000x5000),
                    so you almost always want to resize for U-Net.
        augment:    apply flips, rotations, intensity jitter (train only)
        cache_resolved_paths: precompute resolved DICOM paths in __init__ to
                    avoid directory scanning every epoch.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        image_size: tuple = (512, 512),
        augment: bool = False,
        cache_resolved_paths: bool = True,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment

        # Combine multiple ROIs for the same image? CBIS-DDSM has one row per
        # abnormality, so a single mammogram may appear multiple times with
        # different masks. We keep them as separate samples here — union them
        # in a collate_fn if you'd rather have one mask per image.

        if cache_resolved_paths:
            self._resolve_all()
        else:
            self._resolved = None

    def _resolve_all(self):
        """Pre-resolve DICOM paths and drop rows with missing files."""
        keep = []
        resolved = []
        for i, row in self.df.iterrows():
            img = resolve_dicom(self.data_root, row["image file path"])
            msk = resolve_dicom(self.data_root, row["ROI mask file path"])
            crp = resolve_dicom(self.data_root, row["cropped image file path"])
            if img is None or msk is None:
                continue
            keep.append(i)
            resolved.append((img, msk, crp))
        self.df = self.df.loc[keep].reset_index(drop=True)
        self._resolved = resolved
        print(f"[CBISDDSMSegDataset] {len(self.df)} samples after resolving paths.")

    def __len__(self):
        return len(self.df)

    def _get_paths(self, idx):
        if self._resolved is not None:
            return self._resolved[idx]
        row = self.df.iloc[idx]
        return (
            resolve_dicom(self.data_root, row["image file path"]),
            resolve_dicom(self.data_root, row["ROI mask file path"]),
            resolve_dicom(self.data_root, row["cropped image file path"]),
        )

    def __getitem__(self, idx):
        img_path, msk_path, crp_path = self._get_paths(idx)

        image = load_dicom(img_path)
        mask_raw = load_dicom(msk_path)
        crop_raw = load_dicom(crp_path) if crp_path is not None else None
        mask = pick_mask(mask_raw, crop_raw, image.shape)
        mask = (mask > 0.5).astype(np.float32)

        # To tensors, add channel dim
        image_t = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        # Resize (bilinear for image, nearest for mask to preserve binary labels)
        image_t = TF.resize(image_t, self.image_size, antialias=True)
        mask_t = TF.resize(mask_t, self.image_size,
                           interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)

        # Binarize again in case interpolation introduced anything
        mask_t = (mask_t > 0.5).float()

        # Pathology label (useful for multi-task or just logging)
        label = 1 if self.df.iloc[idx]["pathology"] == "MALIGNANT" else 0

        return {
            "image": image_t,
            "mask": mask_t,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": self.df.iloc[idx]["patient_id"],
        }

    @staticmethod
    def _augment(image, mask):
        # Horizontal flip (left/right breast — valid augmentation)
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Vertical flip (less anatomically motivated but common in medseg)
        if random.random() < 0.3:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # Small rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle,
                              interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle,
                             interpolation=TF.InterpolationMode.NEAREST)
        # Brightness/contrast jitter on the image only
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = image.clamp(0.0, 1.0)
        return image, mask