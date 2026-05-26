"""MobileNet glaucoma classifiers — Paper 2 model variants (Esengönül & Cunha, 2023).

Three model variants (controlled by ``model_variant``):

Model-1   MobileNet (frozen) → GAP → Dense(32, ReLU) → Dropout
          → Dense(32, ReLU) → Dropout → Dense(num_classes, sigmoid)

Model-2   MobileNet (frozen) → GAP → Dense(256, ReLU) → Dropout
          → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout
          → Dense(32, ReLU) → Dropout → Dense(num_classes, sigmoid)

Model-FT  Fully trainable MobileNet + same head as Model-1.

All variants use Adam optimiser and cross-entropy loss (≡ categorical_crossentropy).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
    MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score,
)

MODEL_VARIANTS = ("model1", "model2", "model_ft")


class MobileNetPaper2(L.LightningModule):
    """MobileNet with configurable dense head for glaucoma classification.

    Args:
        model_variant:   One of ``"model1"``, ``"model2"``, ``"model_ft"``.
        num_classes:     2 for binary, 3 for Harvard-style (early/advanced/none).
        pretrained:      Load ImageNet weights.
        lr:              Learning rate for Adam.
        dropout:         Dropout probability for head layers.
        freeze_backbone: Freeze MobileNet backbone (overridden to False for model_ft).
    """

    def __init__(
        self,
        model_variant: str = "model1",
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}, got '{model_variant}'")

        self.backbone = timm.create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            num_classes=0,  # keep GAP only, returns [B, features]
        )
        in_features: int = self.backbone.num_features

        # Model-FT is always fully trainable; others freeze backbone
        should_freeze = freeze_backbone and (model_variant != "model_ft")
        if should_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if model_variant in ("model1", "model_ft"):
            self.head = nn.Sequential(
                nn.Linear(in_features, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 32),          nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, num_classes),
            )
        else:  # model2
            self.head = nn.Sequential(
                nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32),           nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, num_classes),
            )

        is_binary = (num_classes == 2)
        if is_binary:
            self.train_auc        = BinaryAUROC()
            self.val_auc          = BinaryAUROC()
            self.val_acc          = BinaryAccuracy()
            self.val_f1           = BinaryF1Score()
            self.val_sensitivity  = BinaryRecall()
            self.val_specificity  = BinarySpecificity()
            self.test_auc         = BinaryAUROC()
            self.test_acc         = BinaryAccuracy()
            self.test_f1          = BinaryF1Score()
            self.test_sensitivity = BinaryRecall()
            self.test_specificity = BinarySpecificity()
        else:
            self.train_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
            self.val_auc   = MulticlassAUROC(num_classes=num_classes, average="macro")
            self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")
            self.test_auc  = MulticlassAUROC(num_classes=num_classes, average="macro")
            self.test_acc  = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self.test_f1   = MulticlassF1Score(num_classes=num_classes, average="macro")
            self.val_sensitivity  = None
            self.val_specificity  = None
            self.test_sensitivity = None
            self.test_specificity = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.hparams.num_classes == 2:
            return torch.softmax(logits, dim=-1)[:, 1]
        return torch.softmax(logits, dim=-1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"])
        probs = self._probs(logits)
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"])
        probs = self._probs(logits)
        preds = logits.argmax(dim=-1)
        self.val_auc.update(probs, batch["label"])
        self.val_acc.update(preds, batch["label"])
        self.val_f1.update(preds, batch["label"])
        if self.val_sensitivity is not None:
            self.val_sensitivity.update(preds, batch["label"])
            self.val_specificity.update(preds, batch["label"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc.compute(), sync_dist=True)
        self.log("val_f1",  self.val_f1.compute(),  sync_dist=True)
        if self.val_sensitivity is not None:
            self.log("val_sensitivity", self.val_sensitivity.compute(), sync_dist=True)
            self.log("val_specificity", self.val_specificity.compute(), sync_dist=True)
            self.val_sensitivity.reset()
            self.val_specificity.reset()
        self.val_auc.reset(); self.val_acc.reset(); self.val_f1.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        probs = self._probs(logits)
        preds = logits.argmax(dim=-1)
        self.test_auc.update(probs, batch["label"])
        self.test_acc.update(preds, batch["label"])
        self.test_f1.update(preds, batch["label"])
        if self.test_sensitivity is not None:
            self.test_sensitivity.update(preds, batch["label"])
            self.test_specificity.update(preds, batch["label"])

    def on_test_epoch_end(self) -> None:
        self.log("test_auc", self.test_auc.compute(), prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), sync_dist=True)
        self.log("test_f1",  self.test_f1.compute(),  sync_dist=True)
        if self.test_sensitivity is not None:
            self.log("test_sensitivity", self.test_sensitivity.compute(), sync_dist=True)
            self.log("test_specificity", self.test_specificity.compute(), sync_dist=True)
            self.test_sensitivity.reset()
            self.test_specificity.reset()
        self.test_auc.reset(); self.test_acc.reset(); self.test_f1.reset()

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
