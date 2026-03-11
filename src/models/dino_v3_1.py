"""DINOv3 glaucoma classification model — version 1."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity



class _Head(nn.Module):
    """Three-layer MLP classification head with LayerNorm.

    Architecture (per layer):
        LayerNorm -> Linear -> GELU -> (Dropout)
    Final layer is a plain Linear projection to num_classes.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DinoV3_1(L.LightningModule):
    """Glaucoma binary classifier backed by a DINOv2 timm model.

    Args:
        backbone_name: timm model identifier, e.g. ``"vit_small_patch16_dinov3"``.
        pretrained: Whether to load ImageNet-pretrained weights.
        hidden_dim: Width of the first hidden layer of the head.
        num_classes: Number of output classes (default 2 for binary classification).
        dropout: Dropout probability used inside the head.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
    """

    def __init__(
        self,
        backbone_name: str = "vit_huge_plus_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 896,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Backbone — features only, no classifier head.
        # img_size must be passed so timm interpolates the positional embeddings
        # from the pretrained 224×224 grid to the new resolution. Without it,
        # forward passes at img_size != 224 will raise a shape mismatch error.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            img_size=img_size,
        )
        embed_dim: int = self.backbone.num_features

        self.head = _Head(
            in_features=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_sensitivity = BinaryRecall()
        self.val_specificity = BinarySpecificity()
        self.test_auc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_sensitivity = BinaryRecall()
        self.test_specificity = BinarySpecificity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["label"])

        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, batch["label"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["label"])

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        self.val_auc.update(probs, batch["label"])
        self.val_acc.update(preds, batch["label"])
        self.val_f1.update(preds, batch["label"])
        self.val_sensitivity.update(preds, batch["label"])
        self.val_specificity.update(preds, batch["label"])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("val_sensitivity", self.val_sensitivity.compute(), prog_bar=True, sync_dist=True)
        self.log("val_specificity", self.val_specificity.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_sensitivity.reset()
        self.val_specificity.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        self.test_auc.update(probs, batch["label"])
        self.test_acc.update(preds, batch["label"])
        self.test_f1.update(preds, batch["label"])
        self.test_sensitivity.update(preds, batch["label"])
        self.test_specificity.update(preds, batch["label"])

    def on_test_epoch_end(self) -> None:
        self.log("test_auc", self.test_auc.compute(), prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("test_sensitivity", self.test_sensitivity.compute(), prog_bar=True, sync_dist=True)
        self.log("test_specificity", self.test_specificity.compute(), prog_bar=True, sync_dist=True)
        self.test_auc.reset()
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_sensitivity.reset()
        self.test_specificity.reset()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
