"""Fundus Train/Val Data Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FundusTrainValDataset(Dataset):
    """Fundus Train/Val Data dataset for glaucoma classification.

    The dataset is organized with Train/ and Validation/ folders, each containing
    Glaucoma_Negative/ and Glaucoma_Positive/ subdirectories.

    Args:
        data_dir: Path to the root data directory containing Fundus_Train_Val_Data folder.
        split: Dataset split, either "train" or "validation".
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "Fundus_Train_Val_Data" / "Fundus_Scanes_Sorted"
        self.transforms = transforms

        # Normalize split name
        split_lower = split.lower()
        if split_lower in ("train", "training"):
            split_dir = "Train"
        elif split_lower in ("val", "validation"):
            split_dir = "Validation"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'validation'.")

        self.split_dir = self.data_dir / split_dir
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Collect images from both positive and negative folders
        self.image_paths = []
        self.labels = []

        neg_dir = self.split_dir / "Glaucoma_Negative"
        pos_dir = self.split_dir / "Glaucoma_Positive"

        if neg_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                neg_images = sorted(neg_dir.glob(ext))
                self.image_paths.extend(neg_images)
                self.labels.extend([0] * len(neg_images))

        if pos_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                pos_images = sorted(pos_dir.glob(ext))
                self.image_paths.extend(pos_images)
                self.labels.extend([1] * len(pos_images))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = np.array(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }
