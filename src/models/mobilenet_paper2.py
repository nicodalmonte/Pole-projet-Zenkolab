"""MobileNet-V2 glaucoma classifiers — Paper 2 model variants.

Three model variants (controlled by ``model_variant``):

Model-1
    MobileNetV2 (frozen) → GAP → Dense(32, ReLU) → Dropout
    → Dense(32, ReLU) → Dropout → Dense(2)

Model-2
    MobileNetV2 (frozen) → GAP → Dense(256, ReLU) → Dropout
    → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout
    → Dense(32, ReLU) → Dropout → Dense(2)

Model-FT  (fine-tuned)
    Same head as Model-1, but backbone fully trainable from the start.

All variants use:
    - Adam optimiser, cross-entropy loss
    - ReduceLROnPlateau scheduler (patience=3)
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)

MODEL_VARIANTS = ("model1", "model2", "model_ft")


class MobileNetPaper2(L.LightningModule):
    """MobileNet-V2 with configurable dense head for binary glaucoma classification.

    Args:
        model_variant:   One of ``"model1"``, ``"model2"``, ``"model_ft"``.
        pretrained:      Load ImageNet weights.
        lr:              Learning rate for Adam.
        weight_decay:    L2 regularisation coefficient.
        dropout:         Dropout probability for head layers.
        class_weights:   [w_neg, w_pos] for imbalanced datasets.
        freeze_backbone: Freeze backbone (overridden to False for model_ft).
    """

    def __init__(
        self,
        model_variant: str = "model1",
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.5,
        class_weights: list[float] | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}, got '{model_variant}'")

        self.backbone = timm.create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            num_classes=0,  # keep GAP, remove classifier
        )
        in_features: int = self.backbone.num_features  # 1280

        # Model-FT is always fully trainable; Model-1 and Model-2 freeze backbone
        should_freeze = freeze_backbone and (model_variant != "model_ft")
        if should_freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if model_variant in ("model1", "model_ft"):
            self.head = nn.Sequential(
                nn.Linear(in_features, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 32),          nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 2),
            )
        else:  # model2
            self.head = nn.Sequential(
                nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32),           nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 2),
            )

        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None
        )
        self.register_buffer("loss_weight", weight)

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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["label"], weight=self.loss_weight)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

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
        self.log("val_auc",         self.val_auc.compute(),         prog_bar=True, sync_dist=True)
        self.log("val_acc",         self.val_acc.compute(),                        sync_dist=True)
        self.log("val_f1",          self.val_f1.compute(),                         sync_dist=True)
        self.log("val_sensitivity", self.val_sensitivity.compute(),                sync_dist=True)
        self.log("val_specificity", self.val_specificity.compute(),                sync_dist=True)
        self.val_auc.reset(); self.val_acc.reset(); self.val_f1.reset()
        self.val_sensitivity.reset(); self.val_specificity.reset()

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
        self.log("test_auc",         self.test_auc.compute(),         prog_bar=True, sync_dist=True)
        self.log("test_acc",         self.test_acc.compute(),                        sync_dist=True)
        self.log("test_f1",          self.test_f1.compute(),                         sync_dist=True)
        self.log("test_sensitivity", self.test_sensitivity.compute(),                sync_dist=True)
        self.log("test_specificity", self.test_specificity.compute(),                sync_dist=True)
        self.test_auc.reset(); self.test_acc.reset(); self.test_f1.reset()
        self.test_sensitivity.reset(); self.test_specificity.reset()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
