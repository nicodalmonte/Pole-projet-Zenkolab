from __future__ import annotations

import lightning as L
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    BinarySpecificity,
    Recall,
)


class DinoV3V1(L.LightningModule):
    """Binary glaucoma classifier using a timm DINOv3 encoder and a 2-layer dense head."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        # Two dense layers for binary prediction.
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")
        self.train_specificity = BinarySpecificity()
        self.train_sensibility = Recall(task="binary")

        self.val_auc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_specificity = BinarySpecificity()
        self.val_sensibility = Recall(task="binary")

        self.test_auc = AUROC(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_specificity = BinarySpecificity()
        self.test_sensibility = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features).squeeze(1)
        return logits

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch["image"]
        y = batch["label"].float()

        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits)
        y_int = y.long()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "train":
            self.train_auc.update(probs, y_int)
            self.train_acc.update(probs, y_int)
            self.train_specificity.update(probs, y_int)
            self.train_sensibility.update(probs, y_int)
        elif stage == "val":
            self.val_auc.update(probs, y_int)
            self.val_acc.update(probs, y_int)
            self.val_specificity.update(probs, y_int)
            self.val_sensibility.update(probs, y_int)
        else:
            self.test_auc.update(probs, y_int)
            self.test_acc.update(probs, y_int)
            self.test_specificity.update(probs, y_int)
            self.test_sensibility.update(probs, y_int)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True)
        self.log("train_accuracy", self.train_acc.compute(), prog_bar=True)
        self.log("train_specificity", self.train_specificity.compute(), prog_bar=False)
        self.log("train_sensibility", self.train_sensibility.compute(), prog_bar=False)
        self.train_auc.reset()
        self.train_acc.reset()
        self.train_specificity.reset()
        self.train_sensibility.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.log("val_accuracy", self.val_acc.compute(), prog_bar=True)
        self.log("val_specificity", self.val_specificity.compute(), prog_bar=False)
        self.log("val_sensibility", self.val_sensibility.compute(), prog_bar=False)
        self.val_auc.reset()
        self.val_acc.reset()
        self.val_specificity.reset()
        self.val_sensibility.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test_auc", self.test_auc.compute(), prog_bar=True)
        self.log("test_accuracy", self.test_acc.compute(), prog_bar=True)
        self.log("test_specificity", self.test_specificity.compute(), prog_bar=False)
        self.log("test_sensibility", self.test_sensibility.compute(), prog_bar=False)
        self.test_auc.reset()
        self.test_acc.reset()
        self.test_specificity.reset()
        self.test_sensibility.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }