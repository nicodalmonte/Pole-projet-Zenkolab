"""Distillation losses.

Three components
----------------
1. L_KD       — KL-divergence on softened logits (Hinton et al., T²-scaled)
2. L_angular  — ArcFace margin loss on the student embedding
3. L_feat     — Weighted MSE on intermediate features (adaptive multi-teacher)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. KD loss
# ---------------------------------------------------------------------------

def kd_loss(
    logits_s: torch.Tensor,       # (B, C)  — student raw logits (no margin)
    logits_t_avg: torch.Tensor,   # (B, C)  — weighted-average teacher logits
    temperature: float = 4.0,
) -> torch.Tensor:
    """T²-scaled KL divergence between softened student and teacher distributions."""
    T = temperature
    p_s = F.log_softmax(logits_s    / T, dim=-1)
    p_t = F.softmax(logits_t_avg    / T, dim=-1)
    return T * T * F.kl_div(p_s, p_t, reduction="batchmean")


# ---------------------------------------------------------------------------
# 2. ArcFace loss
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """Additive angular margin loss (ArcFace / InsightFace).

    The weight matrix W is used as class prototypes in the embedding space.
    For the KD loss, call ``cosine_logits(feat)`` to get scale*cos(θ) without
    the margin penalty.

    Args:
        embed_dim : Dimension of the input feature vector.
        num_classes: Number of output classes.
        margin    : Additive angular margin m (radians).
        scale     : Feature scale factor s.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 2,
        margin: float = 0.3,
        scale: float = 32.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute cos(m) and sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # threshold for safe formula
        self.mm    = math.sin(math.pi - margin) * margin

    def cosine_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Return scale * cos(θ_c) for each class, without margin. Used for KD."""
        cos_theta = F.linear(F.normalize(feat), F.normalize(self.weight))
        return self.scale * cos_theta

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace cross-entropy loss."""
        cos_theta = F.linear(F.normalize(feat), F.normalize(self.weight))  # (B, C)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=1e-8))

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Safe fall-back for θ > π − m  (avoid cos going negative)
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm,
        )

        # Inject margin only on the true class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        return F.cross_entropy(self.scale * logits, labels)


# ---------------------------------------------------------------------------
# 3. Feature alignment loss (adaptive multi-teacher)
# ---------------------------------------------------------------------------

def feature_loss(
    proj_feats_s: list[torch.Tensor],   # K × (B, d_t)  — projected student per layer
    feats_t_list: list[list[torch.Tensor]],  # K × T × (B, d_t)  — teacher per layer
) -> torch.Tensor:
    """Adaptive multi-teacher feature alignment loss.

    For each selected student layer k and each teacher j, compute:
      - ω_j^(k) = softmax_j( cosine_similarity(proj_s_k, feat_t_j_k) )
      - L_feat   = Σ_k Σ_j ω_j^(k) · MSE(proj_s_k, feat_t_j_k)

    Args:
        proj_feats_s   : List of length K. Each entry is (B, d_t_j) — projected
                         student features for layer k, already projected to the
                         teacher's dimension.  One tensor per (k, j) pair is
                         expected, so the outer list is K and the projector
                         outputs are stacked per-teacher outside this function.
        feats_t_list   : List of length K (student layers). Each sub-list has T
                         tensors (one per teacher), shape (B, d_t).

    Note: ``proj_feats_s[k]`` is actually a list of T projected tensors too
    (one per teacher, potentially different target dims per teacher).
    We therefore accept the same K × T structure for both.
    """
    total = torch.tensor(0.0, device=proj_feats_s[0][0].device)
    K = len(proj_feats_s)

    for k in range(K):
        proj_s_k = proj_feats_s[k]   # list of T tensors (B, d_t_j)
        feats_t_k = feats_t_list[k]  # list of T tensors (B, d_t_j)
        T_teachers = len(feats_t_k)

        # Cosine similarity scores for adaptive weights (on normalised features)
        scores = []
        for j in range(T_teachers):
            sim = F.cosine_similarity(
                F.normalize(proj_s_k[j].detach(), dim=-1),
                F.normalize(feats_t_k[j].detach(), dim=-1),
                dim=-1,
            ).mean()
            scores.append(sim)

        weights = F.softmax(torch.stack(scores), dim=0)  # (T,)

        # Weighted MSE on L2-normalised features (scale-invariant)
        for j in range(T_teachers):
            mse = F.mse_loss(
                F.normalize(proj_s_k[j], dim=-1),
                F.normalize(feats_t_k[j], dim=-1),
            )
            total = total + weights[j] * mse

    return total
