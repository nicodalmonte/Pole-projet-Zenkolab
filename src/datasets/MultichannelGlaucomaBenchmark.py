"""Multichannel Glaucoma Benchmark Dataset.

Compilation of multiple fundus datasets by deathtrooper (Kaggle).
Source: deathtrooper/multichannel-glaucoma-benchmark-dataset

Known duplicates with existing datasets (excluded by default):
  - ORIGA         → ORIGADataset
  - G1020         → G1020Dataset
  - EyePACS-Glaucoma → AIROGSLightDataset (same EyePACS source)
  - REFUGE1-train / REFUGE1-val → REFUGE2Dataset (predecessor challenge)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DUPLICATE_SOURCES = {
    "ORIGA",
    "G1020",
    "EyePACS-Glaucoma",
    "REFUGE1-train",
    "REFUGE1-val",
}


def _source_of(name: str) -> str:
    return re.sub(r"-\d+$", "", name)


class MultichannelGlaucomaBenchmarkDataset(Dataset):
    """Multichannel Glaucoma Benchmark – full-fundus images.

    Args:
        data_dir: Root data directory containing the MultichannelGlaucoma folder.
        sources: Subset of source dataset names to include. None = all non-duplicate sources.
        exclude_duplicates: When sources is None, skip the sub-datasets already covered
            by other Dataset classes in this project.
        transforms: Optional torchvision transforms.
    """

    ALL_SOURCES = {
        "BEH", "CRFO-v4", "DR-HAGIS", "DRISHTI-GS1-test", "DRISHTI-GS1-train",
        "EyePACS-Glaucoma", "FIVES", "G1020", "HRF", "JSIEC-1000", "LES-AV",
        "OIA-ODIR-TEST-OFFLINE", "OIA-ODIR-TEST-ONLINE", "OIA-ODIR-TRAIN",
        "ORIGA", "PAPILA", "REFUGE1-train", "REFUGE1-val", "sjchoi86-HRF",
    }

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        sources: list[str] | None = None,
        exclude_duplicates: bool = True,
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir) / "MultichannelGlaucoma"
        self.image_dir = self.data_dir / "images"
        self.csv_path = self.data_dir / "metadata.csv"
        self.transforms = transforms

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")

        if sources is None:
            active = self.ALL_SOURCES - (DUPLICATE_SOURCES if exclude_duplicates else set())
        else:
            active = set(sources)

        self.samples: list[tuple[Path, int]] = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = _source_of(row["names"])
                if src not in active:
                    continue
                label = int(row["types"])
                if label == -1:  # unknown / unlabeled
                    continue
                # fundus column gives the relative path inside the dataset
                img_path = self.image_dir / Path(row["names"] + ".png").name
                if not img_path.exists():
                    # try the fundus column
                    fundus_rel = row.get("fundus", "").lstrip("/")
                    img_path = self.data_dir / fundus_rel if fundus_rel else img_path
                if img_path.exists():
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found. active_sources={active}"
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
