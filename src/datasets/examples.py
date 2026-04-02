"""Example usage of the glaucoma classification datasets."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import ACRIMADataset, FundusTrainValDataset, LAGDataset, ORIGADataset, REFUGE2Dataset, G1020Dataset, RIMONEDataset


def get_basic_transforms(image_size: int = 224) -> dict:
    """Get basic image transforms for training and evaluation.

    Args:
        image_size: Target image size.

    Returns:
        Dictionary with 'train' and 'eval' transforms.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {"train": train_transform, "eval": eval_transform}


def example_single_dataset(dataset_class, split: str = "train", name: str = "Dataset"):
    """Example of loading a single dataset."""
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print(f"{'='*60}")

    try:
        # Without transforms
        dataset = dataset_class(data_dir="data/datasets", split=split)
        print(f"✓ Loaded {len(dataset)} samples from {name} ({split})")

        # Get first sample
        sample = dataset[0]
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Label: {sample['label'].item()} (0=non-glaucoma, 1=glaucoma)")
        print(f"  - Path: {Path(sample['path']).name}")

        # With transforms
        transforms_dict = get_basic_transforms(image_size=224)
        dataset_with_transforms = dataset_class(
            data_dir="data/datasets",
            split=split,
            transforms=transforms_dict["train"],
        )

        sample_transformed = dataset_with_transforms[0]
        print(f"  - Transformed image shape: {sample_transformed['image'].shape}")

    except Exception as e:
        print(f"✗ Error loading {name}: {e}")


def example_dataloader(dataset_class, split: str = "train", name: str = "Dataset"):
    """Example of creating a DataLoader from a dataset."""
    print(f"\n{'='*60}")
    print(f"DataLoader Example: {name}")
    print(f"{'='*60}")

    try:
        transforms_dict = get_basic_transforms(image_size=224)
        dataset = dataset_class(
            data_dir="data/datasets",
            split=split,
            transforms=transforms_dict["train"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Set to > 0 for multi-process loading
            pin_memory=torch.cuda.is_available(),
        )

        # Get first batch
        batch = next(iter(dataloader))
        print(f"✓ Created DataLoader from {name}")
        print(f"  - Batch size: {batch['image'].shape[0]}")
        print(f"  - Image shape per sample: {batch['image'].shape[1:]}")
        print(f"  - Labels in batch: {batch['label'].tolist()}")
        print(f"  - Label distribution: {torch.bincount(batch['label']).tolist()}")

    except Exception as e:
        print(f"✗ Error creating DataLoader for {name}: {e}")


def main():
    """Run all dataset examples."""
    print("\n" + "=" * 60)
    print("GLAUCOMA CLASSIFICATION DATASETS - USAGE EXAMPLES")
    print("=" * 60)

    # ACRIMA Dataset
    example_single_dataset(ACRIMADataset, split="train", name="ACRIMA")
    example_dataloader(ACRIMADataset, split="train", name="ACRIMA")

    # Fundus Train/Val Data
    example_single_dataset(
        FundusTrainValDataset, split="train", name="Fundus (Train)"
    )
    example_dataloader(
        FundusTrainValDataset, split="train", name="Fundus (Train)"
    )

    example_single_dataset(
        FundusTrainValDataset, split="validation", name="Fundus (Validation)"
    )

    # LAG Dataset
    example_single_dataset(LAGDataset, split="train", name="LAG (Train)")
    example_dataloader(LAGDataset, split="train", name="LAG (Train)")

    example_single_dataset(
        LAGDataset, split="validation", name="LAG (Validation)"
    )

    # ORIGA Dataset
    example_single_dataset(ORIGADataset, split="train", name="ORIGA")
    example_dataloader(ORIGADataset, split="train", name="ORIGA")

    # REFUGE2 Dataset
    example_single_dataset(REFUGE2Dataset, split="train", name="REFUGE2 (Train)")
    example_dataloader(REFUGE2Dataset, split="train", name="REFUGE2 (Train)")

    example_single_dataset(REFUGE2Dataset, split="val", name="REFUGE2 (Val)")
    example_single_dataset(REFUGE2Dataset, split="test", name="REFUGE2 (Test)")

    # G1020 Dataset
    example_single_dataset(G1020Dataset, split="train", name="G1020 (Train)")
    example_dataloader(G1020Dataset, split="train", name="G1020 (Train)")
    example_single_dataset(G1020Dataset, split="test", name="G1020 (Test)")

    # RIM-ONE Dataset
    example_single_dataset(RIMONEDataset, split="train", name="RIM-ONE (Train)")
    example_dataloader(RIMONEDataset, split="train", name="RIM-ONE (Train)")
    example_single_dataset(RIMONEDataset, split="test", name="RIM-ONE (Test)")

    print(f"\n{'='*60}")
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
