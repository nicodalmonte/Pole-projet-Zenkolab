"""Distillation model (Lightning module).

Architecture
------------
Student : DINOv3-Small  (embed=384,  depth=12)
Teachers: EVA02-Large   (embed=1024, depth=24)
          DINOv3-Large  (embed=1024, depth=24)

Image size: 448 px for all models.

Phase 1 — Feature alignment warm-up
    Only the student backbone and projectors are trained.
    Loss: L_feat (weighted MSE on K selected intermediate layers).

Phase 2 — Full distillation
    All student parameters + projectors + ArcFace head are trained.
    Loss: α·L_KD + β·L_angular + γ·L_feat
    α, β are ramped up linearly during a warm-up period; γ starts high
    then converges to its final value.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)

from src.distillation.loss import ArcFaceLoss, kd_loss, feature_loss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDENT_BACKBONE   = "vit_small_patch16_dinov3.lvd1689m"
STUDENT_EMBED_DIM  = 384
STUDENT_DEPTH      = 12

TEACHER_SPECS = {
    "eva02_large": {
        "backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "embed_dim": 1024,
        "depth": 24,
    },
    "dinov3_large": {
        "backbone": "vit_large_patch16_dinov3.lvd1689m",
        "embed_dim": 1024,
        "depth": 24,
    },
}

# Student block indices (0-based) to align with teacher intermediate features.
STUDENT_ALIGN_LAYERS = [2, 5, 8, 11]


# ---------------------------------------------------------------------------
# Projectors  (Linear + LayerNorm, one per student_layer × teacher pair)
# ---------------------------------------------------------------------------

class _Projector(nn.Module):
    """Linear projection + LayerNorm: d_s → d_t."""

    def __init__(self, d_s: int, d_t: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_s, d_t),
            nn.LayerNorm(d_t),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Teacher wrapper — registers hooks, extracts intermediate + final features
# ---------------------------------------------------------------------------

class TeacherWithHooks(nn.Module):
    """Wraps a frozen timm backbone + classification head.

    Hooks are registered on `backbone.blocks[layer_idx]` to capture
    intermediate token sequences.  Features are mean-pooled over tokens
    (including CLS / register tokens) to give a fixed (B, d) vector.

    Args:
        backbone    : Frozen timm model (num_classes=0, global_pool="avg").
        head        : Linear head on top of backbone.
        align_layers: Teacher block indices to hook (mapped proportionally from
                      the student's STUDENT_ALIGN_LAYERS).
        embed_dim   : Backbone output dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        align_layers: list[int],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.backbone     = backbone
        self.head         = head
        self.align_layers = align_layers
        self.embed_dim    = embed_dim

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        self._feats: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_idx in self.align_layers:
            block = self.backbone.blocks[layer_idx]

            def _hook(module, inp, out, idx=layer_idx):
                # out may be tensor (B, N, d) or tuple; take first element
                t = out[0] if isinstance(out, (tuple, list)) else out
                # Mean-pool over token dimension → (B, d)
                self._feats[idx] = t.mean(dim=1)

            self._hooks.append(block.register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Return (logits (B,C), intermediate_feats {layer_idx: (B,d)})."""
        self._feats = {}
        features = self.backbone(x)   # (B, d) — global_pool="avg"
        logits   = self.head(features)
        return logits, dict(self._feats)


# ---------------------------------------------------------------------------
# Helper: build teacher layer mapping
# ---------------------------------------------------------------------------

def _map_student_layers_to_teacher(
    student_layers: list[int],
    student_depth:  int,
    teacher_depth:  int,
) -> list[int]:
    """Proportionally map student block indices to teacher block indices."""
    result = []
    for k in student_layers:
        t_idx = round(k / (student_depth - 1) * (teacher_depth - 1))
        result.append(min(t_idx, teacher_depth - 1))
    return result


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DistillationModule(L.LightningModule):
    """Multi-teacher distillation with two training phases.

    Phase 1 (epochs 0 … phase1_epochs-1):
        Only projectors + student backbone are optimised.
        Loss = L_feat (feature alignment only).

    Phase 2 (epochs phase1_epochs … max_epochs):
        Full optimisation of student + projectors + ArcFace head.
        Loss = α·L_KD + β·L_angular + γ·L_feat
        α, β are linearly warmed up over warmup_epochs_p2.
    """

    def __init__(
        self,
        # teacher checkpoint paths
        ckpt_eva02_large:  str = "",
        ckpt_dinov3_large: str = "",
        # image / model config
        image_size: int = 448,
        num_classes: int = 2,
        # optimisation
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        # distillation hyper-params
        temperature:  float = 4.0,
        arcface_margin: float = 0.3,
        arcface_scale:  float = 32.0,
        # loss weights (final values)
        alpha: float = 1.0,   # L_KD
        beta:  float = 1.0,   # L_angular
        gamma: float = 1.0,   # L_feat
        # phase scheduling
        phase1_epochs:    int = 5,   # warm-up (feat only)
        warmup_epochs_p2: int = 5,   # ramp-up α, β in phase 2
        # class imbalance
        class_weights: list[float] | None = None,
        # infrastructure
        unfreeze_backbone_epoch: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── Student backbone (DINOv3-Small) ─────────────────────────────────
        self.student_backbone = timm.create_model(
            STUDENT_BACKBONE,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self._student_frozen = False
        if unfreeze_backbone_epoch > 0:
            for p in self.student_backbone.parameters():
                p.requires_grad = False
            self._student_frozen = True

        # ── ArcFace classification head ──────────────────────────────────────
        self.arcface = ArcFaceLoss(
            embed_dim=STUDENT_EMBED_DIM,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        # ── Teachers ─────────────────────────────────────────────────────────
        self.teachers = nn.ModuleDict()
        ckpt_map = {
            "eva02_large":  ckpt_eva02_large,
            "dinov3_large": ckpt_dinov3_large,
        }
        self.teacher_align_layers: dict[str, list[int]] = {}

        for name, spec in TEACHER_SPECS.items():
            t_align = _map_student_layers_to_teacher(
                STUDENT_ALIGN_LAYERS, STUDENT_DEPTH, spec["depth"]
            )
            self.teacher_align_layers[name] = t_align

            backbone = timm.create_model(
                spec["backbone"],
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            head = nn.Sequential(
                nn.LayerNorm(spec["embed_dim"]),
                nn.Linear(spec["embed_dim"], num_classes),
            )

            ckpt_path = ckpt_map[name]
            if ckpt_path:
                state = torch.load(ckpt_path, map_location="cpu")
                # Lightning checkpoint — extract state_dict
                if "state_dict" in state:
                    sd = state["state_dict"]
                    bb_sd  = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
                    hd_sd  = {k[len("head.layers."):]: v for k, v in sd.items() if k.startswith("head.layers.")}
                    backbone.load_state_dict(bb_sd, strict=False)
                    head[1].weight.data = hd_sd.get("2.weight", head[1].weight.data)
                    head[1].bias.data   = hd_sd.get("2.bias",   head[1].bias.data)
                else:
                    backbone.load_state_dict(state, strict=False)
                print(f"[teacher] {name} loaded from {ckpt_path}")
            else:
                print(f"[teacher] {name}: no checkpoint — using pretrained weights from timm")
                backbone = timm.create_model(
                    spec["backbone"], pretrained=True, num_classes=0, global_pool="avg"
                )

            self.teachers[name] = TeacherWithHooks(
                backbone=backbone,
                head=head,
                align_layers=t_align,
                embed_dim=spec["embed_dim"],
            )

        # ── Projectors: one per (student_align_layer, teacher) ──────────────
        # projectors[teacher_name][str(student_layer_idx)] -> _Projector
        self.projectors = nn.ModuleDict()
        for t_name, spec in TEACHER_SPECS.items():
            self.projectors[t_name] = nn.ModuleDict({
                str(k): _Projector(STUDENT_EMBED_DIM, spec["embed_dim"])
                for k in STUDENT_ALIGN_LAYERS
            })

        # ── Student intermediate feature hooks ───────────────────────────────
        self._student_feats: dict[int, torch.Tensor] = {}
        self._student_hooks: list[Any] = []
        self._register_student_hooks()

        # ── Metrics ──────────────────────────────────────────────────────────
        self.val_auc  = BinaryAUROC()
        self.val_acc  = BinaryAccuracy()
        self.val_f1   = BinaryF1Score()
        self.val_sens = BinaryRecall()
        self.val_spec = BinarySpecificity()
        self.test_auc  = BinaryAUROC()
        self.test_acc  = BinaryAccuracy()
        self.test_f1   = BinaryF1Score()
        self.test_sens = BinaryRecall()
        self.test_spec = BinarySpecificity()

        # ── Class weights ────────────────────────────────────────────────────
        cw = class_weights or [1.0, 1.0]
        self.register_buffer("loss_weight", torch.tensor(cw, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Student hooks
    # ------------------------------------------------------------------

    def _register_student_hooks(self) -> None:
        for layer_idx in STUDENT_ALIGN_LAYERS:
            block = self.student_backbone.blocks[layer_idx]

            def _hook(module, inp, out, idx=layer_idx):
                t = out[0] if isinstance(out, (tuple, list)) else out
                self._student_feats[idx] = t.mean(dim=1)   # (B, d_s)

            self._student_hooks.append(block.register_forward_hook(_hook))

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @property
    def _current_phase(self) -> int:
        """1 = feature warm-up, 2 = full distillation."""
        return 1 if self.current_epoch < self.hparams.phase1_epochs else 2

    def _loss_weights(self) -> tuple[float, float, float]:
        """Return (alpha, beta, gamma) for the current epoch."""
        if self._current_phase == 1:
            return 0.0, 0.0, 1.0

        ep_in_p2  = self.current_epoch - self.hparams.phase1_epochs
        wu        = max(self.hparams.warmup_epochs_p2, 1)
        ramp      = min(ep_in_p2 / wu, 1.0)
        alpha = self.hparams.alpha * ramp
        beta  = self.hparams.beta  * ramp
        # gamma starts at 1.0, converges to hparams.gamma
        gamma = 1.0 + (self.hparams.gamma - 1.0) * ramp
        return alpha, beta, gamma

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _student_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        self._student_feats = {}
        feat = self.student_backbone(x)     # (B, d_s)
        return feat, dict(self._student_feats)

    def _compute_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (total_loss, student_probs)."""
        alpha, beta, gamma = self._loss_weights()

        # ── Student forward ──────────────────────────────────────────────────
        feat_s, student_inter = self._student_forward(images)

        # ── ArcFace angular loss (phase 2 only) ──────────────────────────────
        if beta > 0:
            L_angular = self.arcface(feat_s, labels)
        else:
            L_angular = feat_s.new_tensor(0.0)

        # Student logits (no margin) for KD
        logits_s = self.arcface.cosine_logits(feat_s)

        # ── Teacher forward — per-teacher losses ─────────────────────────────
        per_teacher_kd:   dict[str, torch.Tensor] = {}
        per_teacher_feat: dict[str, torch.Tensor] = {}

        for t_name, teacher in self.teachers.items():
            t_logits, t_inter = teacher(images)

            # Per-teacher KD loss
            if alpha > 0:
                per_teacher_kd[t_name] = kd_loss(logits_s, t_logits, self.hparams.temperature)
            else:
                per_teacher_kd[t_name] = feat_s.new_tensor(0.0)

            # Per-teacher feature alignment loss
            if gamma > 0:
                t_align = self.teacher_align_layers[t_name]
                proj_s_k = []
                feat_t_k = []
                for s_layer, t_layer in zip(STUDENT_ALIGN_LAYERS, t_align):
                    s_feat = student_inter[s_layer]
                    t_feat = t_inter[t_layer]
                    proj   = self.projectors[t_name][str(s_layer)](s_feat)
                    proj_s_k.append([proj])
                    feat_t_k.append([t_feat])
                per_teacher_feat[t_name] = feature_loss(proj_s_k, feat_t_k)
            else:
                per_teacher_feat[t_name] = feat_s.new_tensor(0.0)

        # ── Aggregate: mean across teachers ──────────────────────────────────
        L_kd   = torch.stack(list(per_teacher_kd.values())).mean()
        L_feat = torch.stack(list(per_teacher_feat.values())).mean()

        # ── Total loss ───────────────────────────────────────────────────────
        loss = alpha * L_kd + beta * L_angular + gamma * L_feat
        L_ce = F.cross_entropy(logits_s, labels, weight=self.loss_weight)
        loss = loss + L_ce

        probs = F.softmax(logits_s, dim=-1)[:, 1]
        return loss, probs, per_teacher_kd, per_teacher_feat

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        # Unfreeze student backbone after warm-up
        target = int(self.hparams.unfreeze_backbone_epoch)
        if self._student_frozen and self.current_epoch >= target:
            for p in self.student_backbone.parameters():
                p.requires_grad = True
            self._student_frozen = False
            self.print(f"[distil] Student backbone unfrozen at epoch {self.current_epoch}.")

        phase = self._current_phase
        a, b, g = self._loss_weights()
        self.print(f"[distil] Phase {phase} | α={a:.2f} β={b:.2f} γ={g:.2f}")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, per_kd, per_feat = self._compute_loss(batch["image"], batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for t_name in per_kd:
            short = t_name.replace("_large", "").replace("02", "02")  # eva02, dinov3
            self.log(f"kd_{short}",   per_kd[t_name],   on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"feat_{short}", per_feat[t_name],  on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        feat_s, _ = self._student_forward(batch["image"])
        logits    = self.arcface.cosine_logits(feat_s)
        loss      = F.cross_entropy(logits, batch["label"], weight=self.loss_weight)
        probs     = F.softmax(logits, dim=-1)[:, 1]
        preds     = logits.argmax(dim=-1)

        self.val_auc.update(probs, batch["label"])
        self.val_acc.update(preds, batch["label"])
        self.val_f1.update(preds, batch["label"])
        self.val_sens.update(preds, batch["label"])
        self.val_spec.update(preds, batch["label"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc",  self.val_auc.compute(),  prog_bar=True, sync_dist=True)
        self.log("val_acc",  self.val_acc.compute(),  prog_bar=True, sync_dist=True)
        self.log("val_f1",   self.val_f1.compute(),   prog_bar=True, sync_dist=True)
        self.log("val_sens", self.val_sens.compute(),  prog_bar=True, sync_dist=True)
        self.log("val_spec", self.val_spec.compute(),  prog_bar=True, sync_dist=True)
        for m in (self.val_auc, self.val_acc, self.val_f1, self.val_sens, self.val_spec):
            m.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        feat_s, _ = self._student_forward(batch["image"])
        logits    = self.arcface.cosine_logits(feat_s)
        probs     = F.softmax(logits, dim=-1)[:, 1]
        preds     = logits.argmax(dim=-1)
        self.test_auc.update(probs,  batch["label"])
        self.test_acc.update(preds,  batch["label"])
        self.test_f1.update(preds,   batch["label"])
        self.test_sens.update(preds, batch["label"])
        self.test_spec.update(preds, batch["label"])

    def on_test_epoch_end(self) -> None:
        self.log("test_auc",  self.test_auc.compute(),  prog_bar=True, sync_dist=True)
        self.log("test_acc",  self.test_acc.compute(),  prog_bar=True, sync_dist=True)
        self.log("test_f1",   self.test_f1.compute(),   prog_bar=True, sync_dist=True)
        self.log("test_sens", self.test_sens.compute(),  prog_bar=True, sync_dist=True)
        self.log("test_spec", self.test_spec.compute(),  prog_bar=True, sync_dist=True)
        for m in (self.test_auc, self.test_acc, self.test_f1, self.test_sens, self.test_spec):
            m.reset()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Teachers are frozen — only student + projectors + arcface are optimised
        params = (
            list(self.student_backbone.parameters())
            + list(self.projectors.parameters())
            + list(self.arcface.parameters())
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }
