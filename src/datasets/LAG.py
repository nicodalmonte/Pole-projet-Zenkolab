"""LAG Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LAGDataset(Dataset):
    """LAG (Large Always Glaucoma?) dataset for glaucoma classification.

    The dataset is organized with train/, validation/, and test/ folders.
    Within each folder, images are prefixed with 'g.' (glaucoma) or 'ng.' (non-glaucoma).

    Args:
        data_dir: Path to the root data directory containing LAG folder.
        split: Dataset split, one of "train", "validation", or "test".
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "LAG"
        self.transforms = transforms

        # Normalize split name
        split_lower = split.lower()
        if split_lower == "train":
            self.split_dir = self.data_dir / "train"
        elif split_lower in ("val", "validation"):
            self.split_dir = self.data_dir / "validation"
        elif split_lower == "test":
            self.split_dir = self.data_dir / "test"
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', or 'test'."
            )

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Collect all images and assign labels based on prefix
        self.image_paths = []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.image_paths.extend(sorted(self.split_dir.glob(ext)))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Label is determined by prefix: "g." = glaucoma (1), "ng." = non-glaucoma (0)
        filename = image_path.name
        label = 1 if filename.startswith("g.") else 0

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
