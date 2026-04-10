"""RETFound DINOv2 glaucoma classification model (Lightning)."""

from __future__ import annotations

import json

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from huggingface_hub import hf_hub_download
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity


def _infer_timm_arch(cfg: dict) -> str:
    """Derive a timm model name from a HuggingFace config dict."""
    hidden = cfg.get("hidden_size", 768)
    patch = cfg.get("patch_size", 16)
    size = "large" if hidden >= 1024 else "base" if hidden >= 768 else "small"
    return f"vit_{size}_patch{patch}_224"


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


class RETFoundDinoV2(L.LightningModule):
    """Glaucoma binary classifier backed by RETFound DINOv2."""

    def __init__(
        self,
        backbone_name: str = "hf_hub:YukunZhou/RETFound_dinov2_meh",
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

        if backbone_name.startswith("hf_hub:"):
            hf_model_id = backbone_name[len("hf_hub:"):]
            config_path = hf_hub_download(repo_id=hf_model_id, filename="config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            timm_arch = cfg.get("architecture") or _infer_timm_arch(cfg)
            self.backbone = timm.create_model(
                timm_arch,
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )
            if pretrained:
                weights_name = hf_model_id.split("/")[-1] + ".pth"
                weights_path = hf_hub_download(repo_id=hf_model_id, filename=weights_name)
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[RETFoundDinoV2] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[RETFoundDinoV2] Unexpected keys: {len(unexpected)}")
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
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
