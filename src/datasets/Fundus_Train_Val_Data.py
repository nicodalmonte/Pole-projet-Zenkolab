import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class FundusTrainValDataset(Dataset):
    """
    Dataset wrapper for Fundus_Train_Val_Data/Fundus_Scanes_Sorted.

    Expected structure:
    Fundus_Scanes_Sorted/
      Train/
        Glaucoma_Negative/
        Glaucoma_Positive/
      Validation/
        Glaucoma_Negative/
        Glaucoma_Positive/
    """

    def __init__(
        self,
        data_dir: str = "data/datasets",
        split: str = "train",
        transforms=None,
    ):
        self.data_dir = Path(data_dir) / "Fundus_Train_Val_Data" / "Fundus_Scanes_Sorted"
        self.transforms = transforms

        split_key = split.lower()
        split_map = {
            "train": "Train",
            "val": "Validation",
            "validation": "Validation",
        }
        if split_key not in split_map:
            raise ValueError("split must be one of: train, val, validation")

        split_dir = self.data_dir / split_map[split_key]
        neg_dir = split_dir / "Glaucoma_Negative"
        pos_dir = split_dir / "Glaucoma_Positive"

        self.samples = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.samples.extend((p, 0) for p in sorted(neg_dir.glob(ext)))
            self.samples.extend((p, 1) for p in sorted(pos_dir.glob(ext)))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label = self.samples[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }
