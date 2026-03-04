"""
Glaucoma classification fine-tuning a linear head on top of DINOv3
(vit_large_patch16_dinov3.lvd1689m) using PyTorch Lightning.

Training strategy
-----------------
* Backbone is FROZEN – only the classification head is trained.
  Pass --unfreeze_at_epoch <N> to thaw the backbone at epoch N with a
  10× smaller learning rate (layer-wise fine-tuning).
* Class-imbalance (train split: ~2658 non-glaucoma / ~1226 glaucoma) is
  handled with a weighted cross-entropy loss computed at startup.
* Metrics tracked: accuracy, AUC-ROC, F1 (macro) on val/test.

Usage
-----
python dino_train.py                          # defaults
python dino_train.py --max_epochs 30 --lr 1e-3 --unfreeze_at_epoch 10
python dino_train.py --data_dir data/LAG --batch_size 64 --devices 2

Resume training
---------------
# resume from last auto-saved checkpoint (checkpoints/last.ckpt)
python dino_train.py --max_epochs 40 --resume_last

# resume from an explicit checkpoint path
python dino_train.py --max_epochs 40 --resume_from checkpoints/dinov3-glaucoma-epoch=12-val_auc=0.9123.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import timm
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import AUROC, Accuracy, F1Score
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
IMG_SIZE = 224          # DINOv3 ViT-L default input size
NUM_CLASSES = 2         # glaucoma / non-glaucoma


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(split: str) -> transforms.Compose:
    """Return augmentation pipeline for train / val-test."""
    mean = (0.5, 0.5, 0.5)
    std  = (0.225, 0.225, 0.225)

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class LAGDataModule(L.LightningDataModule):
    """
    DataModule for the LAG (Large-scale Attention-based Glaucoma) dataset.

    Expected folder structure
    -------------------------
    <data_dir>/
        train/
            glaucoma/image/*.jpg
            non_glaucoma/image/*.jpg
        test/
            glaucoma/image/*.jpg
            non_glaucoma/image/*.jpg
    """

    def __init__(
        self,
        data_dir: str = "data/LAG",
        batch_size: int = 32,
        num_workers: int = 8,
        use_weighted_sampler: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        self.train_ds = ImageFolder(
            root=self.data_dir / "train",
            transform=build_transforms("train"),
        )
        self.val_ds = ImageFolder(
            root=self.data_dir / "test",   # use test split as validation
            transform=build_transforms("val"),
        )
        self.test_ds = self.val_ds          # same split for final test

        # Class names: ImageFolder sorts alphabetically
        # glaucoma=0, non_glaucoma=1  (or reverse – handled via labels)
        print(f"[DataModule] Classes: {self.train_ds.classes}")
        print(f"[DataModule] Train: {len(self.train_ds)} samples, "
              f"Val/Test: {len(self.val_ds)} samples")

    # ------------------------------------------------------------------
    def _build_sampler(self) -> WeightedRandomSampler:
        targets = torch.tensor(self.train_ds.targets)
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[targets]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        sampler = self._build_sampler() if self.use_weighted_sampler else None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    # ------------------------------------------------------------------
    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted CE loss (CPU tensor)."""
        targets = torch.tensor(self.train_ds.targets)
        counts = torch.bincount(targets).float()
        weights = counts.sum() / (len(counts) * counts)
        return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlaucomaClassifier(L.LightningModule):
    """
    DINOv3 ViT-L backbone + linear classification head.

    Parameters
    ----------
    num_classes      Number of output classes (2 for glaucoma/non-glaucoma).
    lr               Learning rate for the head (and backbone when unfrozen).
    weight_decay     AdamW weight decay.
    class_weights    1-D tensor of per-class weights for the CE loss.
    unfreeze_at_epoch
                     If > 0, the backbone is thawed at this epoch with lr/10.
    label_smoothing  Label smoothing for cross-entropy.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        unfreeze_at_epoch: int = 0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # ---- Backbone -------------------------------------------------------
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,   # remove classifier head → raw features
        )
        embed_dim: int = self.backbone.num_features

        # Freeze backbone by default
        self._set_backbone_grad(requires_grad=False)

        # ---- Classification head --------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, num_classes),
        )

        # ---- Loss -----------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        # ---- Metrics --------------------------------------------------------
        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc  = Accuracy(**metric_kwargs)
        self.val_acc    = Accuracy(**metric_kwargs)
        self.val_auc    = AUROC(task="multiclass", num_classes=num_classes)
        self.val_f1     = F1Score(task="multiclass", num_classes=num_classes,
                                  average="macro")
        self.test_acc   = Accuracy(**metric_kwargs)
        self.test_auc   = AUROC(task="multiclass", num_classes=num_classes)
        self.test_f1    = F1Score(task="multiclass", num_classes=num_classes,
                                  average="macro")

        self._backbone_unfrozen = False

    # ------------------------------------------------------------------
    def _set_backbone_grad(self, requires_grad: bool):
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    # ------------------------------------------------------------------
    def on_train_epoch_start(self):
        epoch = self.current_epoch
        if (
            self.hparams.unfreeze_at_epoch > 0
            and epoch >= self.hparams.unfreeze_at_epoch
            and not self._backbone_unfrozen
        ):
            self._set_backbone_grad(requires_grad=True)
            self._backbone_unfrozen = True
            # Rebuild optimizer with two param groups
            self.trainer.strategy.setup_optimizers(self.trainer)
            print(f"[Epoch {epoch}] Backbone unfrozen – LR={self.hparams.lr / 10:.2e}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    # ------------------------------------------------------------------
    def _shared_step(self, batch, stage: str):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        return loss, probs, labels

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch, "train")
        self.train_acc(probs, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch, "val")
        self.val_acc(probs, labels)
        self.val_auc(probs, labels)
        self.val_f1(probs, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  self.val_acc,  prog_bar=True)
        self.log("val/auc",  self.val_auc,  prog_bar=True)
        self.log("val/f1",   self.val_f1,   prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch, "test")
        self.test_acc(probs, labels)
        self.test_auc(probs, labels)
        self.test_f1(probs, labels)
        self.log("test/loss", loss)
        self.log("test/acc",  self.test_acc)
        self.log("test/auc",  self.test_auc)
        self.log("test/f1",   self.test_f1)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self._backbone_unfrozen:
            param_groups = [
                {"params": self.backbone.parameters(),
                 "lr": self.hparams.lr / 10},
                {"params": self.head.parameters(),
                 "lr": self.hparams.lr},
            ]
        else:
            param_groups = [
                {"params": self.head.parameters(),
                 "lr": self.hparams.lr},
            ]

        optimizer = torch.optim.Adam(
            param_groups,
        )
        return {
            "optimizer": optimizer,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a DINOv3 head for glaucoma classification."
    )
    p.add_argument("--data_dir",          default="data/LAG",  type=str)
    p.add_argument("--batch_size",        default=32,           type=int)
    p.add_argument("--num_workers",       default=8,            type=int)
    p.add_argument("--max_epochs",        default=20,           type=int)
    p.add_argument("--lr",                default=1e-3,         type=float)
    p.add_argument("--weight_decay",      default=1e-4,         type=float)
    p.add_argument("--label_smoothing",   default=0.1,          type=float)
    p.add_argument("--unfreeze_at_epoch", default=0,            type=int,
                   help="Epoch at which to unfreeze backbone (0 = never).")
    p.add_argument("--devices",           default=1,            type=int,
                   help="Number of GPUs to use.")
    p.add_argument("--precision",         default="16-mixed",   type=str,
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="Floating-point precision.")
    p.add_argument("--ckpt_dir",          default="checkpoints", type=str)
    p.add_argument("--no_weighted_sampler", action="store_true",
                   help="Disable WeightedRandomSampler (use class-weighted loss only).")
    p.add_argument("--seed",              default=42,            type=int)
    # ----- Resume -----------------------------------------------------------
    resume = p.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume_from",
        default=None,
        type=str,
        metavar="CKPT_PATH",
        help="Resume training from this checkpoint file.",
    )
    resume.add_argument(
        "--resume_last",
        action="store_true",
        help="Resume from <ckpt_dir>/last.ckpt (auto-detected).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    # ---- Data ---------------------------------------------------------------
    dm = LAGDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    dm.setup()
    class_weights = dm.class_weights
    print(f"[Main] Class weights: {class_weights}")

    # ---- Model --------------------------------------------------------------
    model = GlaucomaClassifier(
        num_classes=NUM_CLASSES,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        label_smoothing=args.label_smoothing,
    )

    # ---- Callbacks ----------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="dinov3-glaucoma-{epoch:02d}-{val/auc:.4f}",
            monitor="val/auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/auc",
            patience=7,
            mode="max",
            verbose=True,
        ),
    ]

    # ---- Trainer ------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,    # faster when False; set True for full reproducibility
    )

    # ---- Resolve resume checkpoint ------------------------------------------
    ckpt_path: str | None = None
    if args.resume_from:
        ckpt_path = args.resume_from
        print(f"[Main] Resuming from: {ckpt_path}")
    elif args.resume_last:
        last = Path(args.ckpt_dir) / "last.ckpt"
        if last.exists():
            ckpt_path = str(last)
            print(f"[Main] Resuming from last checkpoint: {ckpt_path}")
        else:
            print(f"[Main] WARNING: --resume_last specified but {last} not found. "
                  "Starting from scratch.")

    # ---- Train + Test -------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
