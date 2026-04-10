"""Lightweight student-only eval module.

Loads only the student backbone + ArcFace head from a DistillationModule
checkpoint, without instantiating the teacher models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)

from src.distillation.loss import ArcFaceLoss
from src.distillation.model import STUDENT_BACKBONE, STUDENT_EMBED_DIM


class StudentEval(L.LightningModule):
    """Inference-only wrapper around the distilled student backbone + ArcFace head.

    Instantiated from a DistillationModule checkpoint via ``from_distillation_ckpt``.
    No teacher weights are ever loaded.
    """

    def __init__(self, num_classes: int = 2, image_size: int = 448, threshold: float = 0.5) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Store as backbone_name so build_transforms works in the test harness
        self.hparams.backbone_name = STUDENT_BACKBONE
        self.hparams.img_size = image_size
        self.threshold = threshold

        self.backbone = timm.create_model(
            STUDENT_BACKBONE,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.arcface = ArcFaceLoss(
            embed_dim=STUDENT_EMBED_DIM,
            num_classes=num_classes,
        )

        for split in ("test",):
            setattr(self, f"{split}_auc",  BinaryAUROC())
            setattr(self, f"{split}_acc",  BinaryAccuracy())
            setattr(self, f"{split}_f1",   BinaryF1Score())
            setattr(self, f"{split}_sens", BinaryRecall())
            setattr(self, f"{split}_spec", BinarySpecificity())

    @classmethod
    def from_distillation_ckpt(cls, ckpt_path: str) -> "StudentEval":
        """Build a StudentEval by extracting student weights from a distillation checkpoint."""
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        hp = state.get("hyper_parameters", {})

        num_classes = int(hp.get("num_classes", 2))
        image_size  = int(hp.get("image_size",  448))

        model = cls(num_classes=num_classes, image_size=image_size, threshold=0.5)

        bb_sd = {k[len("student_backbone."):]: v
                 for k, v in sd.items() if k.startswith("student_backbone.")}
        arc_sd = {k[len("arcface."):]: v
                  for k, v in sd.items() if k.startswith("arcface.")}

        missing_bb  = model.backbone.load_state_dict(bb_sd,  strict=True)
        missing_arc = model.arcface.load_state_dict(arc_sd,  strict=True)
        print(f"[student_eval] backbone  — missing: {missing_bb.missing_keys}")
        print(f"[student_eval] arcface   — missing: {missing_arc.missing_keys}")
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.arcface.cosine_logits(feat)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["image"])
        probs  = F.softmax(logits, dim=-1)[:, 1]
        preds  = (probs >= self.threshold).long()
        self.test_auc.update(probs,  batch["label"])
        self.test_acc.update(preds,  batch["label"])
        self.test_f1.update(preds,   batch["label"])
        self.test_sens.update(preds, batch["label"])
        self.test_spec.update(preds, batch["label"])

    def on_test_epoch_end(self) -> None:
        self.log("test_auc",  self.test_auc.compute(),  prog_bar=True)
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=True)
        self.log("test_f1",   self.test_f1.compute(),   prog_bar=True)
        self.log("test_sens", self.test_sens.compute(), prog_bar=True)
        self.log("test_spec", self.test_spec.compute(), prog_bar=True)
        for m in (self.test_auc, self.test_acc, self.test_f1, self.test_sens, self.test_spec):
            m.reset()

    def configure_optimizers(self):
        return None
