"""ACRIMA Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ACRIMADataset(Dataset):
    """ACRIMA dataset for glaucoma classification.

    Images without "_g" suffix are non-glaucoma (label 0).
    Images with "_g" suffix are glaucoma (label 1).

    Args:
        data_dir: Path to the root data directory containing ACRIMA folder.
        split: Dataset split. ACRIMA doesn't have explicit splits, so any value is accepted.
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "ACRIMA"
        self.image_dir = self.data_dir / "Images"
        self.transforms = transforms

        if not self.image_dir.exists():
            raise FileNotFoundError(f"ACRIMA Images directory not found: {self.image_dir}")

        # Collect all image paths
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.image_paths.extend(sorted(self.image_dir.glob(ext)))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Label is determined by presence of "_g" in filename
        label = 1 if "_g_" in image_path.name else 0

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = np.array(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }
