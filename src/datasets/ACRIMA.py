import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class ACRIMADataset(Dataset):
    """
    ACRIMA dataset for binary glaucoma classification.

    Label convention in filenames:
    - files containing "_g_" -> glaucoma (1)
    - all other files -> non-glaucoma (0)
    """

    def __init__(
        self,
        data_dir: str = "data/datasets",
        split: str = "train",
        transforms=None,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        self.data_dir = Path(data_dir) / "ACRIMA" / "Images"
        self.split = split.lower()
        self.transforms = transforms

        all_images = sorted(self.data_dir.glob("*.jpg")) + sorted(self.data_dir.glob("*.JPG"))
        if not all_images:
            raise FileNotFoundError(f"No images found in {self.data_dir}")

        train_ratio, val_ratio, test_ratio = split_ratios
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
            raise ValueError("split_ratios must sum to 1.0")

        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if self.split == "train":
            self.image_paths = all_images[:n_train]
        elif self.split in {"val", "validation"}:
            self.image_paths = all_images[n_train:n_train + n_val]
        elif self.split == "test":
            self.image_paths = all_images[n_train + n_val:]
        elif self.split == "all":
            self.image_paths = all_images
        else:
            raise ValueError("split must be one of: train, val, test, all")

        if not self.image_paths:
            raise FileNotFoundError(f"No images found for split '{self.split}' in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if "_g_" in image_path.stem.lower() else 0

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }
