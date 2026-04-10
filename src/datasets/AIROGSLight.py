"""AIROGS-Light Dataset for glaucoma classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AIROGSLightDataset(Dataset):
    """AIROGS-Light (EyePACS) dataset for glaucoma classification.

    The dataset is organized in class subfolders under a split folder:
        AIROGSLight/
            eyepac-light-v2-512-jpg/
                train/
                    RG/   (referable glaucoma)
                    NRG/  (non-referable glaucoma)

    This dataset only contains a train split. The split parameter is accepted
    for API consistency but "val"/"validation"/"test" are all mapped to "train".

    Class labels:
        - 0: non-referable glaucoma (NRG/)
        - 1: referable glaucoma     (RG/)

    Args:
        data_dir: Path to the root data directory containing AIROGSLight folder.
        split: Ignored (only train split available). Kept for API consistency.
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = (
            Path(data_dir) / "AIROGSLight" / "eyepac-light-v2-512-jpg" / "train"
        )
        self.transforms = transforms

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"AIROGSLight train directory not found: {self.data_dir}"
            )

        self.image_paths = []
        self.labels = []

        label_map = {
            "RG": 1,
            "NRG": 0,
        }

        for folder_name, label in label_map.items():
            class_dir = self.data_dir / folder_name
            if not class_dir.exists():
                continue

            class_images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                class_images.extend(sorted(class_dir.glob(ext)))

            self.image_paths.extend(class_images)
            self.labels.extend([label] * len(class_images))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_dir}")

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
