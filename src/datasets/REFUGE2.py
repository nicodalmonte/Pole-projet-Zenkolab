import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np

class REFUGE2Dataset(Dataset):
    """
    REFUGE2 Dataset for PyTorch Lightning.
    
    Args:
        data_dir (str): Path to the dataset directory (data/datasets)
        split (str): 'train', 'val', or 'test'
        transforms (callable): Optional image transformations
    """
    
    def __init__(
        self,
        data_dir: str = "data/datasets",
        split: str = "train",
        transforms=None
    ):
        self.data_dir = Path(data_dir) / "REFUGE2"
        self.split = split
        self.transforms = transforms
        
        # Load image paths and labels
        self.image_paths = sorted(
            (self.data_dir / split / "images").glob("*.jpg")
        )
        self.label_dir = self.data_dir / split / "labels"
        
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.data_dir / split / 'images'}"
            )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label/mask if exists
        label_path = self.label_dir / f"{image_path.stem}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) if label_path.exists() else None
        
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            "image": image,
            "label": torch.from_numpy(label).long() if label is not None else None,
            "path": str(image_path)
        }