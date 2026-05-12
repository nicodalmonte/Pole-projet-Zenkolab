from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

class _StudentBackbone(nn.Module):
    """Backbone of the student.

    Args:
        backbone_name: timm backbone for the student.
        pretrained: load ImageNet pretrained weights.
        img_size: input image size.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        img_size: int,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _StudentHead(nn.Module):
    """Small but expressive classification head.

    Architecture:
        LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear
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
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StudentGlaucomaDistilled(L.LightningModule):
    """Compact student for binary glaucoma classification.

    Intended to be trained with a combination of:
    - hard-label CE loss
    - soft-label KD loss from a teacher

    Args:
        backbone_name: timm backbone for the student.
        pretrained: load ImageNet pretrained weights.
        hidden_dim: head hidden size.
        num_classes: normally 2.
        dropout: dropout in the head.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        img_size: input image size.
        class_weights: optional class weights [w_neg, w_pos].
        teacher_model: optional pretrained teacher model.
        kd_alpha: weight on distillation loss.
        kd_temperature: softmax temperature for distillation.
        freeze_backbone_epochs: keep student backbone frozen for first N epochs.
    """

    def __init__(
        self,
        backbone_name: str = "fastvit_t8.apple_dist_in1k",
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 896,
        class_weights: list[float] | None = None,
        teacher_model: nn.Module | None = None,
        kd_alpha: float = 0.7,
        kd_temperature: float = 4.0,
        freeze_backbone_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["teacher_model"])

        self.student = _StudentBackbone(
            backbone_name,
            pretrained=pretrained,
            img_size=img_size,
        )
        embed_dim: int = self.student.backbone.num_features

        self.head = _StudentHead(
            in_features=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.register_buffer("loss_weight", weight)

        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self._student_backbone_is_frozen = False
        if freeze_backbone_epochs > 0:
            self._freeze_backbone()

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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        for p in self.student.backbone.parameters():
            p.requires_grad = False
        self._student_backbone_is_frozen = True

    def _unfreeze_backbone(self) -> None:
        for p in self.student.backbone.parameters():
            p.requires_grad = True
        self._student_backbone_is_frozen = False

    def on_train_epoch_start(self) -> None:
        if self._student_backbone_is_frozen and self.current_epoch >= int(self.hparams.freeze_backbone_epochs):
            self._unfreeze_backbone()
            self.print(f"Unfroze student backbone at epoch {self.current_epoch}.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.student(x)
        return self.head(features)

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL distillation loss with temperature scaling."""
        T = float(self.hparams.kd_temperature)

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # Multiply by T^2 as in standard distillation.
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total_loss, hard_loss, soft_loss)."""
        hard_loss = F.cross_entropy(student_logits, labels, weight=self.loss_weight)

        if self.teacher is None:
            soft_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
            total_loss = hard_loss
            return total_loss, hard_loss, soft_loss

        with torch.no_grad():
            self.teacher.eval()
            teacher_logits = self.teacher(images)

        soft_loss = self._kd_loss(student_logits, teacher_logits)
        alpha = float(self.hparams.kd_alpha)

        total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
        return total_loss, hard_loss, soft_loss

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss, hard_loss, soft_loss = self._compute_loss(logits, labels, images)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_hard_loss", hard_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_soft_loss", soft_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss, hard_loss, soft_loss = self._compute_loss(logits, labels, images)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        self.val_auc.update(probs, labels)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_specificity.update(preds, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_hard_loss", hard_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_soft_loss", soft_loss, on_epoch=True, prog_bar=False, sync_dist=True)

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
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc",
                "interval": "epoch",
            },
        }