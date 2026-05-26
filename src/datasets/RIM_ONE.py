"""RIM-ONE Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RIMONEDataset(Dataset):
    """RIM-ONE dataset for glaucoma classification.

    The dataset is organized as:
    - RIM-ONE_DL_images/
      - partitioned_randomly/ or partitioned_by_hospital/
        - training_set/glaucoma, training_set/normal
        - test_set/glaucoma, test_set/normal

    Class labels:
        - 0: normal (non-glaucoma)
        - 1: glaucoma

    Args:
        data_dir: Path to the root data directory containing RIM-ONE_DL_images.
        split: Dataset split, one of "train" or "test".
            For compatibility, "val"/"validation" are mapped to "test".
        partition: Partition strategy, one of "random" or "hospital".
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        partition: str = "hospital",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "RIM-ONE_DL_images"
        self.transforms = transforms

        partition_lower = partition.lower()
        if partition_lower in ("random", "randomly"):
            partition_dir = "partitioned_randomly"
        elif partition_lower in ("hospital", "by_hospital"):
            partition_dir = "partitioned_by_hospital"
        else:
            raise ValueError(
                f"Invalid partition: {partition}. Must be 'random' or 'hospital'."
            )

        split_lower = split.lower()
        if split_lower in ("train", "training"):
            split_dir = "training_set"
        elif split_lower in ("test", "val", "validation"):
            split_dir = "test_set"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

        self.split_dir = self.data_dir / partition_dir / split_dir
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.image_paths = []
        self.labels = []

        class_dirs = (("normal", 0), ("glaucoma", 1))
        for class_name, label in class_dirs:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue

            class_images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                class_images.extend(sorted(class_dir.glob(ext)))

            self.image_paths.extend(class_images)
            self.labels.extend([label] * len(class_images))

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