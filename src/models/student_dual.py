"""Student model for dual-teacher knowledge distillation (DINO + EVA).

Loss formula
------------
    total = λ_dino * KL(student || teacher_dino)
          + λ_eva  * KL(student || teacher_eva)
          + (1 - λ_dino - λ_eva) * CE(student, hard_labels)

Batch format expected by training_step
---------------------------------------
    "image_a" : student-resolution tensor  (e.g. 224 px — fed to student and DINO teacher)
    "image_b" : EVA-resolution tensor      (e.g. 448 px — fed to EVA teacher only)
    "label"   : ground-truth class indices

Validation / test steps use the standard single-image format ("image", "label").
"""

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
    """Thin wrapper around a timm backbone that strips the classification head."""

    def __init__(self, backbone_name: str, pretrained: bool, img_size: int) -> None:
        super().__init__()
        # Create the timm backbone without the final classification layer (num_classes=0)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, img_size=img_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _StudentHead(nn.Module):
    """Compact classification head: LayerNorm → Dropout → Linear."""

    def __init__(
        self, in_features: int, hidden_dim: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StudentGlaucomaDualDistilled(L.LightningModule):
    """Binary glaucoma student trained with knowledge distillation from two teachers.

    Args:
        backbone_name: timm identifier for the student backbone.
        pretrained: load ImageNet pretrained weights for the student.
        hidden_dim: head hidden size (unused by the current single-layer head, kept for compatibility).
        num_classes: number of output classes (2 for binary).
        dropout: dropout probability in the head.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        img_size: student/DINO-teacher image resolution.
        class_weights: optional [w_neg, w_pos] for the hard CE loss.
        teacher_dino: frozen DINO teacher model (DinoV3_1 Lightning module).
        teacher_eva: frozen EVA teacher model (EvaViT Lightning module).
        kd_lambda_dino: weight λ_dino on the DINO teacher KD loss.
        kd_lambda_eva: weight λ_eva on the EVA teacher KD loss.
        kd_temperature: shared softmax temperature T for both KD losses.
        freeze_backbone_epochs: keep student backbone frozen for first N epochs (0 = never).
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 224,
        class_weights: list[float] | None = None,
        teacher_dino: nn.Module | None = None,
        teacher_eva: nn.Module | None = None,
        kd_lambda_dino: float = 0.4,
        kd_lambda_eva: float = 0.4,
        kd_temperature: float = 4.0,
        freeze_backbone_epochs: int = 0,
    ) -> None:
        super().__init__()
        # Exclude teacher modules from hparams (they are not serialisable Lightning hparams)
        self.save_hyperparameters(ignore=["teacher_dino", "teacher_eva"])

        # Build student encoder and head
        self.student = _StudentBackbone(backbone_name, pretrained=pretrained, img_size=img_size)
        embed_dim: int = self.student.backbone.num_features
        self.head = _StudentHead(
            in_features=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout
        )

        # Optional class-imbalance weighting for the hard CE loss
        weight = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.register_buffer("loss_weight", weight)

        # Store DINO teacher and freeze it permanently
        self.teacher_dino = teacher_dino
        if self.teacher_dino is not None:
            self.teacher_dino.eval()
            for p in self.teacher_dino.parameters():
                p.requires_grad = False

        # Store EVA teacher and freeze it permanently
        self.teacher_eva = teacher_eva
        if self.teacher_eva is not None:
            self.teacher_eva.eval()
            for p in self.teacher_eva.parameters():
                p.requires_grad = False

        # Optionally freeze the student backbone for the first N epochs
        self._student_backbone_is_frozen = False
        if freeze_backbone_epochs > 0:
            self._freeze_backbone()

        # Metrics
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
        # Teachers must stay in eval mode even while the student is training
        if self.teacher_dino is not None:
            self.teacher_dino.eval()
        if self.teacher_eva is not None:
            self.teacher_eva.eval()
        return self

    # ------------------------------------------------------------------
    # Freeze / unfreeze student backbone
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
        # Unfreeze student backbone after the configured warmup period
        if (
            self._student_backbone_is_frozen
            and self.current_epoch >= int(self.hparams.freeze_backbone_epochs)
        ):
            self._unfreeze_backbone()
            self.print(f"Unfroze student backbone at epoch {self.current_epoch}.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Student forward at student resolution (image_a)
        return self.head(self.student(x))

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _kd_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence distillation loss with temperature scaling (Hinton et al., 2015)."""
        T = float(self.hparams.kd_temperature)
        # Log-softmax of student at temperature T
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # Softmax of teacher at temperature T (detached — teacher is already no_grad)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        # Multiply by T² to keep gradient magnitudes consistent across temperatures
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        images_student: torch.Tensor,
        images_eva: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total_loss, hard_loss, kd_loss_dino, kd_loss_eva)."""
        # Hard cross-entropy against ground-truth labels
        hard_loss = F.cross_entropy(student_logits, labels, weight=self.loss_weight)

        lambda_dino = float(self.hparams.kd_lambda_dino)
        lambda_eva = float(self.hparams.kd_lambda_eva)

        # Initialise KD losses to zero in case a teacher is absent
        zero = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        kd_loss_dino = zero
        kd_loss_eva = zero

        # Run both teachers inside no_grad — only teacher logits need no gradient
        with torch.no_grad():
            if self.teacher_dino is not None:
                self.teacher_dino.eval()
                # DINO teacher receives the same resolution images as the student (image_a)
                dino_logits = self.teacher_dino(images_student)

            if self.teacher_eva is not None:
                self.teacher_eva.eval()
                # EVA teacher receives its own higher-resolution images (image_b)
                eva_logits = self.teacher_eva(images_eva)

        # Compute KD losses outside no_grad so gradients flow through student_logits
        if self.teacher_dino is not None:
            kd_loss_dino = self._kd_loss(student_logits, dino_logits)

        if self.teacher_eva is not None:
            kd_loss_eva = self._kd_loss(student_logits, eva_logits)

        # Weighted combination: λ_dino * soft_dino + λ_eva * soft_eva + (1 - λ_dino - λ_eva) * hard
        total_loss = (
            lambda_dino * kd_loss_dino
            + lambda_eva * kd_loss_eva
            + (1.0 - lambda_dino - lambda_eva) * hard_loss
        )
        return total_loss, hard_loss, kd_loss_dino, kd_loss_eva

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # image_a: student/DINO resolution (e.g. 224 px)
        images_student = batch["image_a"]
        # image_b: EVA resolution (e.g. 448 px) — passed only to the EVA teacher
        images_eva = batch["image_b"]
        labels = batch["label"]

        # Forward through student using student-resolution images
        logits = self(images_student)
        total_loss, hard_loss, kd_loss_dino, kd_loss_eva = self._compute_loss(
            logits, labels, images_student, images_eva
        )

        # Positive-class probability for AUC tracking
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, labels)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_hard_loss", hard_loss, on_step=False, on_epoch=True, sync_dist=True)
        # Log individual teacher KD losses for monitoring
        self.log("train_kd_dino", kd_loss_dino, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_kd_eva", kd_loss_eva, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # Validation uses standard single-image batches at student resolution
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        # Validation loss is hard CE only (no teachers involved)
        val_loss = F.cross_entropy(logits, labels, weight=self.loss_weight)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        self.val_auc.update(probs, labels)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_specificity.update(preds, labels)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

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
        # Test uses standard single-image batches at student resolution
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
        # Only optimise student parameters (teachers are frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
