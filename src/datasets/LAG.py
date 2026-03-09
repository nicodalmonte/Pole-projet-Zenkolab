import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class LAGDataset(Dataset):
    """
    LAG dataset where labels are encoded in filename prefixes.

    Filename conventions:
    - g.*   -> glaucoma (1)
    - ng.*  -> non-glaucoma (0)
    """

    def __init__(
        self,
        data_dir: str = "data/datasets",
        split: str = "train",
        transforms=None,
    ):
        self.data_dir = Path(data_dir) / "LAG"
        self.transforms = transforms

        split_key = split.lower()
        split_map = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test",
        }
        if split_key not in split_map:
            raise ValueError("split must be one of: train, val, validation, test")

        split_dir = self.data_dir / split_map[split_key]
        self.image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.JPG"))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stem = image_path.stem.lower()
        if stem.startswith("g."):
            label = 1
        elif stem.startswith("ng."):
            label = 0
        else:
            raise ValueError(f"Unexpected filename format for label inference: {image_path.name}")

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }
