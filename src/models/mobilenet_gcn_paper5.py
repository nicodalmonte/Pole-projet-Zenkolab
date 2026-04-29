"""Paper 5 — MobileNetV2 encoder + Attention Module (AM) + ResGCN for binary
glaucoma classification.

Architecture
------------
  MobileNetV2 (ImageNet pretrained)
    → feature maps  (B, 1280, 7, 7)  at 224×224 input
  Attention Module (5 sequential channel-attention blocks)
    → each block: global channel avg-pool → 1×1 conv → BN → sigmoid → multiply
  Spatial→Graph projection
    → adaptive-avg-pool to 4×4 → flatten → (B, 16, C) nodes
  ResGCN (Residual Graph Convolutional Network)
    → dilated k-NN edge features  (concatenate [xi, xj - xi])
    → 6 residual blocks × ~4 EdgeConv layers  ≈ 24+ edge conv operations
    → vertices grown from 16 → 64 → 256 via upsampling between block groups
  Classification head
    → GAP over nodes → FC(256 → 128, ReLU) → Dropout(0.4) → FC(128 → 2)

Loss: focal loss (γ=2) for classification.

Training strategy (same as all papers):
    Phase 1 : ACRIMA  (backbone frozen, Adam lr=1e-3, 25 epochs)
    Phase 2 : ORIGA   (backbone unfrozen, Adam lr=1e-4, 25 epochs)
    Test    : G1020 + RIM-ONE + REFUGE2 + LAG + JRAIGS + Fundus + AIROGSLight
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary / multi-class focal loss.

    Args:
        gamma:  Focusing parameter (γ). Default 2.
        weight: Class weights tensor (same length as num_classes).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)  # type: ignore[arg-type]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)           # (B, C)
        p     = log_p.exp()                              # (B, C)
        # gather probability of the true class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)
        focal  = -((1 - pt) ** self.gamma) * log_pt                  # (B,)
        if self.weight is not None:
            w = self.weight.gather(0, targets)
            focal = focal * w
        return focal.mean()


# ---------------------------------------------------------------------------
# Attention Module (AM) — channel attention block
# ---------------------------------------------------------------------------

class ChannelAttentionBlock(nn.Module):
    """One channel-attention block.

    global avg-pool → 1×1 conv → BN → sigmoid → channel-wise multiply.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gap   = nn.AdaptiveAvgPool2d(1)           # (B, C, 1, 1)
        self.conv  = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn    = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.bn(self.conv(self.gap(x))))  # (B, C, 1, 1)
        return x * scale


class AttentionModule(nn.Module):
    """Stack of 5 channel-attention blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ChannelAttentionBlock(channels) for _ in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Edge Convolution (EdgeConv)
# ---------------------------------------------------------------------------

def knn_graph(x: torch.Tensor, k: int, dilation: int = 1) -> torch.Tensor:
    """Build dilated k-NN graph indices.

    Args:
        x:        (B, N, C) node features.
        k:        number of neighbours.
        dilation: pick every `dilation`-th neighbour (dilated k-NN).

    Returns:
        idx: (B, N, k) long tensor of neighbour indices.
    """
    B, N, _ = x.shape
    # pairwise squared distance
    inner = -2 * torch.bmm(x, x.transpose(1, 2))          # (B, N, N)
    sq    = (x ** 2).sum(dim=-1, keepdim=True)             # (B, N, 1)
    dist  = sq + inner + sq.transpose(1, 2)                # (B, N, N)  ≥ 0
    # clamp so we never ask for more neighbours than available
    kd    = min(k * dilation, N - 1)
    _, idx_all = dist.topk(kd + 1, dim=-1, largest=False)  # include self → kd+1
    idx_all = idx_all[:, :, 1:]                            # remove self (→ kd entries)
    idx = idx_all[:, :, ::dilation][:, :, :k]             # dilated selection
    return idx.contiguous()


class EdgeConv(nn.Module):
    """Single EdgeConv layer.

    For each node xi and its k neighbours {xj},
    edge feature = [xi ‖ (xj − xi)], then MLP + max-pool over neighbours.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 8, dilation: int = 1) -> None:
        super().__init__()
        self.k        = k
        self.dilation = dilation
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        returns: (B, N, out_ch)
        """
        B, N, C = x.shape
        idx = knn_graph(x, self.k, self.dilation)           # (B, N, k)

        # gather neighbours: (B, N, k, C)
        idx_exp = idx.unsqueeze(-1).expand(B, N, self.k, C)
        x_exp   = x.unsqueeze(2).expand(B, N, self.k, C)
        nbr     = x.unsqueeze(1).expand(B, N, N, C).gather(
            2, idx_exp
        )                                                    # (B, N, k, C)

        edge = torch.cat([x_exp, nbr - x_exp], dim=-1)     # (B, N, k, 2C)
        edge = edge.permute(0, 3, 1, 2)                    # (B, 2C, N, k)
        out  = self.mlp(edge)                               # (B, out_ch, N, k)
        out  = out.max(dim=-1).values                       # (B, out_ch, N)
        return out.permute(0, 2, 1)                         # (B, N, out_ch)


# ---------------------------------------------------------------------------
# Residual EdgeConv block
# ---------------------------------------------------------------------------

class ResEdgeBlock(nn.Module):
    """Two EdgeConv layers with a residual shortcut."""

    def __init__(self, ch: int, k: int = 8, dilation: int = 1) -> None:
        super().__init__()
        self.ec1 = EdgeConv(ch, ch, k=k, dilation=dilation)
        self.ec2 = EdgeConv(ch, ch, k=k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.ec2(self.ec1(x)), 0.2)


# ---------------------------------------------------------------------------
# ResGCN
# ---------------------------------------------------------------------------

class ResGCN(nn.Module):
    """Residual Graph Convolutional Network.

    Nodes are grown across three stages:
        Stage A (16 nodes,  ch=128):  4 residual blocks (dilation 1, 2)
        Stage B (64 nodes,  ch=256):  4 residual blocks (dilation 1, 2)
        Stage C (256 nodes, ch=256):  4 residual blocks (dilation 1, 2)

    Between stages nodes are upsampled via learned linear interpolation.
    """

    def __init__(self, in_ch: int = 1280) -> None:
        super().__init__()
        # project backbone channels to graph node channels
        self.proj = nn.Linear(in_ch, 128)

        # Stage A — 16 nodes, 128 ch
        self.stageA = nn.ModuleList([
            ResEdgeBlock(128, k=8, dilation=d) for d in [1, 2, 1, 2]
        ])

        # A→B: upsample 16→64 nodes, 128→256 ch
        self.upAB   = nn.Linear(128, 256)
        self.up16_64 = nn.Upsample(scale_factor=4, mode="nearest")  # applied on (B,C,N)

        # Stage B — 64 nodes, 256 ch
        self.stageB = nn.ModuleList([
            ResEdgeBlock(256, k=8, dilation=d) for d in [1, 2, 1, 2]
        ])

        # B→C: upsample 64→256 nodes, keep 256 ch
        self.up64_256 = nn.Upsample(scale_factor=4, mode="nearest")

        # Stage C — 256 nodes, 256 ch
        self.stageC = nn.ModuleList([
            ResEdgeBlock(256, k=8, dilation=d) for d in [1, 2, 1, 2]
        ])

    @staticmethod
    def _upsample_nodes(x: torch.Tensor, scale: int) -> torch.Tensor:
        """Nearest-neighbour upsample along the node dimension.
        x: (B, N, C) → (B, N*scale, C)
        """
        return x.repeat_interleave(scale, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_ch) → (B, 256*N', 256)"""
        x = self.proj(x)                                  # (B, 16, 128)

        for blk in self.stageA:
            x = blk(x)                                    # (B, 16, 128)

        # upsample 16→64
        x = self._upsample_nodes(x, 4)                   # (B, 64, 128)
        x = self.upAB(x)                                  # (B, 64, 256)

        for blk in self.stageB:
            x = blk(x)                                    # (B, 64, 256)

        # upsample 64→256
        x = self._upsample_nodes(x, 4)                   # (B, 256, 256)

        for blk in self.stageC:
            x = blk(x)                                    # (B, 256, 256)

        return x                                          # (B, 256, 256)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MobileNetGCNPaper5(L.LightningModule):
    """MobileNetV2 + Attention Module (5 blocks) + ResGCN for binary glaucoma
    classification.

    Args:
        pretrained:      Load ImageNet pretrained weights.
        lr:              Learning rate for Adam.
        weight_decay:    L2 regularisation.
        focal_gamma:     Focal loss focusing parameter (γ). Default 2.
        freeze_backbone: Freeze MobileNetV2 encoder weights (Phase 1).
    """

    def __init__(
        self,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        focal_gamma: float = 2.0,
        freeze_backbone: bool = True,
        lr_patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── Encoder ─────────────────────────────────────────────────────
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mv2 = mobilenet_v2(weights=weights)
        # keep feature extractor, drop classifier
        self.encoder = mv2.features       # → (B, 1280, 7, 7) at 224×224
        enc_ch = 1280

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ── Attention Module ────────────────────────────────────────────
        self.attention = AttentionModule(enc_ch)

        # ── Spatial → Graph (4×4 = 16 nodes) ───────────────────────────
        self.graph_pool = nn.AdaptiveAvgPool2d(4)         # (B, 1280, 4, 4)

        # ── ResGCN ──────────────────────────────────────────────────────
        self.gcn = ResGCN(in_ch=enc_ch)                   # (B, 256, 256)

        # ── Classification head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(gamma=focal_gamma)

        # ── Metrics ─────────────────────────────────────────────────────
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
        feat   = self.encoder(x)                           # (B, 1280, 7, 7)
        feat   = self.attention(feat)                      # (B, 1280, 7, 7)
        nodes  = self.graph_pool(feat)                     # (B, 1280, 4, 4)
        nodes  = nodes.flatten(2).permute(0, 2, 1)        # (B, 16, 1280)
        gcn_out = self.gcn(nodes)                          # (B, 256, 256)
        pooled  = gcn_out.mean(dim=1)                      # (B, 256)  node-wise GAP
        return self.head(pooled)                           # (B, 2)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        logits = self(batch["image"])
        loss   = self.criterion(logits, batch["label"])
        probs  = torch.softmax(logits, -1)[:, 1]
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_auc", self.train_auc.compute(), prog_bar=True, sync_dist=True)
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        logits = self(batch["image"])
        loss   = self.criterion(logits, batch["label"])
        probs  = torch.softmax(logits, -1)[:, 1]
        preds  = logits.argmax(-1)
        self.val_auc.update(probs, batch["label"])
        self.val_acc.update(preds, batch["label"])
        self.val_f1.update(preds, batch["label"])
        self.val_sensitivity.update(preds, batch["label"])
        self.val_specificity.update(preds, batch["label"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_auc",         self.val_auc.compute(),         prog_bar=True, sync_dist=True)
        self.log("val_acc",         self.val_acc.compute(),                        sync_dist=True)
        self.log("val_f1",          self.val_f1.compute(),                         sync_dist=True)
        self.log("val_sensitivity", self.val_sensitivity.compute(),                sync_dist=True)
        self.log("val_specificity", self.val_specificity.compute(),                sync_dist=True)
        self.val_auc.reset(); self.val_acc.reset(); self.val_f1.reset()
        self.val_sensitivity.reset(); self.val_specificity.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch["image"])
        probs  = torch.softmax(logits, -1)[:, 1]
        preds  = logits.argmax(-1)
        self.test_auc.update(probs, batch["label"])
        self.test_acc.update(preds, batch["label"])
        self.test_f1.update(preds, batch["label"])
        self.test_sensitivity.update(preds, batch["label"])
        self.test_specificity.update(preds, batch["label"])

    def on_test_epoch_end(self):
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
        opt = torch.optim.Adam(
            trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=self.hparams.lr_patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_auc", "interval": "epoch"},
        }
