import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class ORIGADataset(Dataset):
    """
    ORIGA dataset wrapper using labels from glaucoma.csv.

    Split strategy from CSV column 'Set':
    - train -> Set A
    - val/test -> Set B
    - all -> all samples
    """

    def __init__(
        self,
        data_dir: str = "data/datasets",
        split: str = "train",
        transforms=None,
    ):
        self.data_dir = Path(data_dir) / "ORIGA"
        self.transforms = transforms

        csv_path = self.data_dir / "glaucoma.csv"
        image_dir = self.data_dir / "ORIGA" / "Images"
        ann_dir = self.data_dir / "ORIGA" / "Semi-automatic-annotations"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing directory: {image_dir}")

        split_key = split.lower()
        if split_key not in {"train", "val", "validation", "test", "all"}:
            raise ValueError("split must be one of: train, val, validation, test, all")

        self.samples = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                set_name = str(row.get("Set", "")).strip().upper()
                include = (
                    split_key == "all"
                    or (split_key == "train" and set_name == "A")
                    or (split_key in {"val", "validation", "test"} and set_name == "B")
                )
                if not include:
                    continue

                filename = row["Filename"]
                image_path = image_dir / filename
                if not image_path.exists():
                    continue

                label = int(row["Glaucoma"])
                ann_path = ann_dir / f"{Path(filename).stem}.mat"

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label": label,
                        "annotation_path": ann_path if ann_path.exists() else None,
                    }
                )

        if not self.samples:
            raise FileNotFoundError("No ORIGA samples found for the requested split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "path": str(image_path),
            "annotation_path": str(sample["annotation_path"]) if sample["annotation_path"] is not None else None,
        }
