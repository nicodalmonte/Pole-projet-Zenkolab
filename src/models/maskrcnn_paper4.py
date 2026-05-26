"""Mask R-CNN backbone (ResNet-50 + FPN) adapted for binary glaucoma classification.

Uses the torchvision Mask R-CNN backbone (ResNet-50 + FPN, pretrained on COCO/ImageNet)
as a feature extractor.  The FPN outputs 256-channel feature maps; we apply global
average pooling on the last FPN level and feed the result to a classification head.

Architecture:
    ResNet-50 + FPN  (Mask R-CNN backbone, COCO pretrained)
    → Global average pooling on last FPN feature map  → (B, 256)
    → Linear(256 → 128, ReLU)
    → Dropout(0.4)
    → Linear(128 → 2)        [cross-entropy]

Training strategy (per paper):
    Phase 1 : ACRIMA  (backbone frozen, Adam lr=1e-3, 25 epochs)
    Phase 2 : ORIGA   (backbone unfrozen, Adam lr=1e-4, 25 epochs)
    Test    : G1020 + all other unseen datasets
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)


class MaskRCNNPaper4(L.LightningModule):
    """ResNet-50 + FPN backbone (Mask R-CNN) for binary glaucoma classification.

    Args:
        pretrained:      Load COCO-pretrained Mask R-CNN weights for the backbone.
        lr:              Learning rate for Adam.
        weight_decay:    L2 regularisation.
        class_weights:   [w_neg, w_pos] for imbalanced datasets.
        freeze_backbone: Freeze backbone weights (Phase 1). Set False for Phase 2.
    """

    def __init__(
        self,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        class_weights: list[float] | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        det_model = maskrcnn_resnet50_fpn(weights=weights)
        self.backbone = det_model.backbone   # BackboneWithFPN: ResNet-50 + FPN
        fpn_out_channels = self.backbone.out_channels  # 256

        self.head = nn.Sequential(
            nn.Linear(fpn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
        features = self.backbone(x)               # OrderedDict of FPN levels
        last = features[list(features.keys())[-1]]  # (B, 256, H', W')
        pooled = F.adaptive_avg_pool2d(last, 1).flatten(1)  # (B, 256)
        return self.head(pooled)

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
        optimizer = torch.optim.Adam(
            trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
