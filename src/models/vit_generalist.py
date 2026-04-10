"""Generalist ViT glaucoma classification model (Lightning)."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity


class _Head(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ViTGeneralist(L.LightningModule):
    """Glaucoma binary classifier backed by a generalist ViT."""

    def __init__(
        self,
        backbone_name: str = "vit_large_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 896,
        class_weights: list[float] | None = None,
        unfreeze_backbone_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )
        embed_dim: int = self.backbone.num_features

        self._backbone_is_frozen = False
        if unfreeze_backbone_epoch > 0:
            self._freeze_backbone()

        self.head = _Head(embed_dim, num_classes, dropout)

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("loss_weight", weight)

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

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_is_frozen = True

    def _unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_is_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"], weight=self.loss_weight)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self) -> None:
        target_epoch = int(self.hparams.unfreeze_backbone_epoch)
        if self._backbone_is_frozen and self.current_epoch >= target_epoch:
            self._unfreeze_backbone()
            self.print(f"Unfroze backbone at epoch {self.current_epoch}.")

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"], weight=self.loss_weight)

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
