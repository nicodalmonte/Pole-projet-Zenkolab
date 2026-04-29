"""EfficientNet-B0 glaucoma classifier — Paper 1 architecture.

    Architecture (per paper):
    EfficientNet-B0 (ImageNet pretrained) — global average pooling built-in
    → Linear(1280 → 128, ReLU)
    → Dropout(0.4)
    → Linear(128 → 1)        [binary cross-entropy]

Training strategy (per paper):
    Phase 1 : Train on ACRIMA       — Adam lr=1e-3, up to 25 epochs
    Phase 2 : Fine-tune on ORIGA    — Adam lr=1e-4 (lower to retain representations)
    Test    : RIM-ONE + all unseen datasets
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


class EfficientNetPaper1(L.LightningModule):
    """EfficientNet-B0 with custom head for binary glaucoma classification.

    Args:
        pretrained:       Load ImageNet weights.
        lr:               Learning rate for Adam.
        weight_decay:     L2 regularisation coefficient.
        class_weights:    [w_neg, w_pos] for imbalanced datasets.
        freeze_backbone:  Freeze backbone weights. Defaults to False to follow paper exactly.
    """

    def __init__(
        self,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        class_weights: list[float] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,   # remove classifier head, keep GAP
        )
        in_features: int = self.backbone.num_features  # 1280

        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if class_weights is not None:
            # For BCEWithLogitsLoss, pos_weight = expected_negatives / expected_positives.
            # Assuming class_weights is [w_neg, w_pos]. We use w_pos / w_neg for pos_weight
            # or just a single scalar if user pre-computed it differently. 
            # We'll use a simple fallback if lists are given:
            pos_w = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32)
        else:
            pos_w = None
        self.register_buffer("pos_weight", pos_w)

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
        return self.head(self.backbone(x)).squeeze(-1)

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = F.binary_cross_entropy_with_logits(
            logits, batch["label"].float(), pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = F.binary_cross_entropy_with_logits(
            logits, batch["label"].float(), pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
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
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
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
        return {"optimizer": optimizer}
