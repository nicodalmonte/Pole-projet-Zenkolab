from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import timm
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.datasets.ACRIMA import ACRIMADataset
from src.datasets.Fundus_Train_Val_Data import FundusTrainValDataset
from src.datasets.LAG import LAGDataset
from src.datasets.ORIGA import ORIGADataset
from src.models.dino_v3_v1 import DinoV3V1


class REFUGE2BinaryTestDataset(Dataset):
    """REFUGE2 test-only dataset converted to binary labels from masks."""

    def __init__(self, data_dir: str = "data/datasets", transforms_fn=None) -> None:
        self.data_dir = Path(data_dir) / "REFUGE2" / "test"
        self.transforms = transforms_fn

        image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "mask"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing REFUGE2 test images dir: {image_dir}")

        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.image_paths.extend(sorted(image_dir.glob(ext)))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for REFUGE2 sample: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        label = int(np.any(mask > 0))

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }


class DinoDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        backbone_name: str,
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.backbone_name = backbone_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        model_cfg = timm.data.resolve_model_data_config(
            timm.create_model(self.backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        )
        model_cfg = dict(model_cfg)
        model_cfg["input_size"] = (3, self.image_size, self.image_size)
        train_transform = timm.data.create_transform(
            **model_cfg,
            is_training=True,
        )
        eval_transform = timm.data.create_transform(
            **model_cfg,
            is_training=False,
        )

        if stage in ("fit", None):
            train_parts = [
                ACRIMADataset(data_dir=self.data_dir, split="train", transforms=train_transform),
                FundusTrainValDataset(data_dir=self.data_dir, split="train", transforms=train_transform),
                LAGDataset(data_dir=self.data_dir, split="train", transforms=train_transform),
                ORIGADataset(data_dir=self.data_dir, split="train", transforms=train_transform),
            ]
            val_parts = [
                ACRIMADataset(data_dir=self.data_dir, split="val", transforms=eval_transform),
                FundusTrainValDataset(data_dir=self.data_dir, split="validation", transforms=eval_transform),
                LAGDataset(data_dir=self.data_dir, split="validation", transforms=eval_transform),
                ORIGADataset(data_dir=self.data_dir, split="val", transforms=eval_transform),
            ]

            self.train_dataset = ConcatDataset(train_parts)
            self.val_dataset = ConcatDataset(val_parts)

        if stage in ("test", None):
            self.test_dataset = REFUGE2BinaryTestDataset(
                data_dir=self.data_dir,
                transforms_fn=eval_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 glaucoma classifier with PyTorch Lightning")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_dinov3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DinoV3V1(
        backbone_name=args.backbone,
        pretrained=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    datamodule = DinoDataModule(
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="dinov3-glaucoma-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()