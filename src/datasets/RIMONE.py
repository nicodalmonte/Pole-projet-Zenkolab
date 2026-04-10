"""RIM-ONE Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RIMONEDataset(Dataset):
    """RIM-ONE DL retinal dataset for glaucoma classification.

    The dataset is organized as:
        RIM-ONE/
            RIM-ONE_DL_images/
                partitioned_by_hospital/
                    train_set/
                        glaucoma/
                        normal/
                    test_set/
                        glaucoma/
                        normal/

    Class labels:
        - 0: non-glaucoma (normal/)
        - 1: glaucoma (glaucoma/)

    Args:
        data_dir: Path to the root data directory containing RIM-ONE folder.
        split: Dataset split, one of "train" or "test".
            For compatibility, "val"/"validation" are mapped to "test".
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "test",
        transforms=None,
    ) -> None:
        self.data_dir = (
            Path(data_dir)
            / "RIM-ONE"
            / "RIM-ONE_DL_images"
            / "partitioned_by_hospital"
        )
        self.transforms = transforms

        split_lower = split.lower()
        if split_lower in ("train", "training"):
            split_dir = "training_set"
        elif split_lower in ("test", "val", "validation"):
            split_dir = "test_set"
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train' or 'test'."
            )

        self.split_dir = self.data_dir / split_dir
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}"
            )

        self.image_paths = []
        self.labels = []

        label_map = {
            "glaucoma": 1,
            "normal": 0,
        }

        for folder_name, label in label_map.items():
            class_dir = self.split_dir / folder_name
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
