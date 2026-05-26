"""Harvard Glaucoma Dataset for glaucoma classification (Kim's Eye Hospital).

Reference: U. Kim, "Machine learn for glaucoma," Harvard Dataverse, 2018.
           https://doi.org/10.7910/DVN/1YRRAC

Structure
---------
    Harvard/
        normal_control/    — label 0 (788 images)
        early_glaucoma/    — label 1 (289 images)
        advanced_glaucoma/ — label 2 (467 images)

Labels
------
    0: normal control
    1: early glaucoma
    2: advanced glaucoma

Total: 1 544 images, 3-class classification (as used in Esengönül & Cunha, 2023).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class HarvardGlaucomaDataset(Dataset):
    """Harvard glaucoma fundus image dataset (3-class).

    Args:
        data_dir:   Path to the root data directory containing the ``Harvard/`` folder.
        transforms: Optional torchvision transforms applied to each image.
    """

    LABEL_MAP = {
        "normal_control":    0,
        "early_glaucoma":    1,
        "advanced_glaucoma": 2,
    }

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        transforms=None,
        # `split` accepted for API consistency but unused (no predefined splits)
        split: str = "train",
    ) -> None:
        self.root = Path(data_dir) / "Harvard"
        self.transforms = transforms

        if not self.root.exists():
            raise FileNotFoundError(
                f"Harvard dataset not found at {self.root}. "
                "Run install_datasets.install(['Harvard']) to download it."
            )

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        for folder, label in self.LABEL_MAP.items():
            class_dir = self.root / folder
            if not class_dir.exists():
                continue
            imgs: list[Path] = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                imgs.extend(sorted(class_dir.glob(ext)))
            self.image_paths.extend(imgs)
            self.labels.extend([label] * len(imgs))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return {
            "image": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "path":  str(self.image_paths[idx]),
        }
