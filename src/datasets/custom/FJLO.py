from src.datasets import FundusTrainValDataset, JRAIGSDataset, LAGDataset, ORIGADataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
import torch
from PIL import Image

class SplitTransformDataset(Dataset):
    """Apply a transform to the image field of a dataset item."""

    def __init__(self, dataset: Dataset, image_transform) -> None:
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            transformed_image = image
        else:
            if hasattr(image, "shape"):
                image = Image.fromarray(image)
            transformed_image = self.image_transform(image)
        sample["image"] = transformed_image
        return sample

def _set_source_name(dataset: Dataset, source_name: str) -> Dataset:
    dataset.source_name = source_name  # type: ignore[attr-defined]
    return dataset

def build_combined_dataset(data_dir: str) -> ConcatDataset:
    """Concatenate every available dataset into one global pool."""
    datasets = [
        _set_source_name(FundusTrainValDataset(data_dir=data_dir, split="train", transforms=None), "Fundus(train)"),
        _set_source_name(FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=None), "Fundus(validation)"),
        _set_source_name(JRAIGSDataset(data_dir=data_dir, transforms=None), "JRAIGS(all)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="train", transforms=None), "LAG(train)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="validation", transforms=None), "LAG(validation)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="test", transforms=None), "LAG(test)"),
        _set_source_name(ORIGADataset(data_dir=data_dir, transforms=None), "ORIGA(all)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="train", partition="hospital", transforms=None), "RIM-ONE(hospital_train)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="test", partition="hospital", transforms=None), "RIM-ONE(hospital_test)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="train", partition="random", transforms=None), "RIM-ONE(random_train)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="test", partition="random", transforms=None), "RIM-ONE(random_test)"),
    ]
    return ConcatDataset(datasets)

def get_datasets(data_dir, train_ratio, val_ratio, test_ratio):
    combined_ds = build_combined_dataset(data_dir)
    total_samples = len(combined_ds)
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_len = int(total_samples * train_ratio / ratio_sum)
    val_len = int(total_samples * val_ratio / ratio_sum)
    test_len = total_samples - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_raw, val_raw, test_raw = random_split(
        combined_ds,
        [train_len, val_len, test_len],
        generator=generator,
    )

    return train_raw, val_raw, test_raw

def FJLODataset(data_dir: str, split="test", transforms=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) -> Dataset:
    """Dataset that concatenates all available datasets and splits them into train/val/test."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")

    train_raw, val_raw, test_raw = get_datasets(data_dir, train_ratio, val_ratio, test_ratio)

    if split == "train":
        return SplitTransformDataset(train_raw, transforms)
    elif split == "val":
        return SplitTransformDataset(val_raw, transforms)
    else:
        return SplitTransformDataset(test_raw, transforms)
    