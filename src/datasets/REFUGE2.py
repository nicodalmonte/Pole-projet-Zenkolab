"""REFUGE2 Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class REFUGE2Dataset(Dataset):
    """REFUGE2 dataset for glaucoma classification.

    Images are organized in train/, val/, and test/ folders with images/subdirectories.
    Labels are determined by file prefix: "g" for glaucoma, "n" for non-glaucoma.

    Args:
        data_dir: Path to the root data directory containing REFUGE2 folder.
        split: Dataset split, one of "train", "val", or "test".
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "REFUGE2"
        self.transforms = transforms

        # Normalize split name
        split_lower = split.lower()
        if split_lower == "train":
            self.image_dir = self.data_dir / "train" / "images"
        elif split_lower in ("val", "validation"):
            self.image_dir = self.data_dir / "val" / "images"
        elif split_lower == "test":
            self.image_dir = self.data_dir / "test" / "images"
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Collect all images
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.image_paths.extend(sorted(self.image_dir.glob(ext)))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Label is determined by prefix: "g" = glaucoma (1), "n" = non-glaucoma (0)
        filename = image_path.name
        label = 1 if filename.startswith("g") else 0

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
