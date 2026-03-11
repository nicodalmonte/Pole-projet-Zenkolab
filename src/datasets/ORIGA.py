"""ORIGA Dataset for glaucoma classification."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ORIGADataset(Dataset):
    """ORIGA (ORIGA-light?) dataset for glaucoma classification.

    The dataset uses a CSV file (glaucoma.csv) to provide labels for images.
    Images are located in the ORIGA/Images/ folder.

    Args:
        data_dir: Path to the root data directory containing ORIGA folder.
        split: Dataset split. ORIGA doesn't have explicit splits; this parameter is ignored
            but kept for API consistency. Can filter based on 'Set' column in CSV if needed.
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "ORIGA"
        self.image_dir = self.data_dir / "ORIGA" / "Images"
        self.csv_file = self.data_dir / "glaucoma.csv"
        self.transforms = transforms

        if not self.image_dir.exists():
            raise FileNotFoundError(f"ORIGA Images directory not found: {self.image_dir}")

        if not self.csv_file.exists():
            raise FileNotFoundError(f"ORIGA CSV file not found: {self.csv_file}")

        # Load labels from CSV
        self.samples = []  # List of tuples: (image_path, label)

        try:
            with open(self.csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("Filename")
                    glaucoma_label = int(row.get("Glaucoma", 0))

                    if filename:
                        image_path = self.image_dir / filename
                        if image_path.exists():
                            self.samples.append((image_path, glaucoma_label))
        except Exception as e:
            raise RuntimeError(f"Failed to load ORIGA CSV: {e}")

        if not self.samples:
            raise FileNotFoundError(
                f"No valid image-label pairs found in {self.csv_file}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label = self.samples[idx]

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
