"""JRAIGS Dataset for glaucoma classification."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class JRAIGSDataset(Dataset):
    """JRAIGS dataset for glaucoma classification.

    The dataset uses a CSV file (filtered_labels.csv) to provide labels for images.
    Images are located in the JRAIGS/images/ folder.

    Labels:
        - 'NRG': Non-glaucoma (label = 0)
        - 'RG': Glaucoma (label = 1)

    Args:
        data_dir: Path to the root data directory containing JRAIGS folder.
        split: Dataset split. JRAIGS doesn't have explicit splits; this parameter is ignored
            but kept for API consistency.
        transforms: Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "JRAIGS"
        self.image_dir = self.data_dir / "images"
        self.csv_file = self.data_dir / "filtered_labels.csv"
        self.transforms = transforms

        if not self.image_dir.exists():
            raise FileNotFoundError(f"JRAIGS images directory not found: {self.image_dir}")

        if not self.csv_file.exists():
            raise FileNotFoundError(f"JRAIGS CSV file not found: {self.csv_file}")

        # Load labels from CSV
        self.samples = []  # List of tuples: (image_path, label)

        try:
            with open(self.csv_file, "r") as f:
                df = pd.read_csv(self.csv_file)
                for _, row in df.iterrows():
                    eye_id = row["Eye ID"]
                    final_label = row["Final Label"]

                    if eye_id and final_label:
                        # Map label: NRG (non-glaucoma) = 0, RG (glaucoma) = 1
                        if final_label == "NRG":
                            label = 0
                        elif final_label == "RG":
                            label = 1
                        else:
                            raise ValueError(f"Unknown label '{final_label}' for Eye ID '{eye_id}' in JRAIGS CSV")

                        # Construct image path
                        image_path = self.image_dir / f"{eye_id}.JPG"
                        if image_path.exists():
                            self.samples.append((image_path, label))
        except Exception as e:
            raise RuntimeError(f"Failed to load JRAIGS CSV: {e}")

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
