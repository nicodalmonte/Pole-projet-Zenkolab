"""
Two model options for glaucoma binary classification:

  1. ViTClassifier    — ViT-Large/16 pretrained on ImageNet-21k (public, fast baseline)
  2. RETFoundClassifier — ViT-Large pretrained on 1.6M retinal images (RETFound MAE)
                         loaded via  timm.create_model("hf_hub:bitfount/RETFound_MAE")

Both expose the same interface: backbone + head, compatible with train.py.
"""

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Shared head builder
# ---------------------------------------------------------------------------
def _make_head(embed_dim: int, num_classes: int, drop_rate: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(drop_rate),
        nn.Linear(embed_dim, embed_dim//2),
        nn.LayerNorm(embed_dim//2),
        nn.Dropout(drop_rate),
        nn.Linear(embed_dim//2, num_classes),
    )


def _print_param_counts(model: nn.Module) -> None:
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total/1e6:.1f}M │ Trainable: {train_/1e6:.1f}M")


# ---------------------------------------------------------------------------
# 1. ViTClassifier — plain ViT-Large, ImageNet-21k (public baseline)
# ---------------------------------------------------------------------------
class ViTClassifier(nn.Module):
    """
    ViT-Large/16 backbone pretrained on ImageNet-21k via timm.
    ~307M parameters. Public weights, no authentication needed.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        drop_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        print("[model] Loading ViT-Large (ImageNet-21k) from timm …")
        self.backbone = timm.create_model(
            "vit_large_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,      # strip head → outputs (B, 1024)
            global_pool="avg",
        )
        self.head = _make_head(self.backbone.embed_dim, num_classes, drop_rate)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_vit(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    drop_rate: float = 0.3,
    load_pretrained: bool = True,
) -> ViTClassifier:
    """Return a ViT-Large classifier with ImageNet-21k weights."""
    model = ViTClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        drop_rate=drop_rate,
        pretrained=load_pretrained,
    )
    _print_param_counts(model)
    return model


# ---------------------------------------------------------------------------
# 2. RETFoundClassifier — ViT-Large pretrained on 1.6M retinal images (MAE)
# ---------------------------------------------------------------------------
class RETFoundClassifier(nn.Module):
    """
    RETFound (ViT-Large MAE) backbone pretrained on 1.6M retinal images.
    Loaded via timm HuggingFace hub shortcut: hf_hub:bitfount/RETFound_MAE
    ~307M parameters. Retinal-domain pretraining → better features for glaucoma.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        drop_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        print("[model] Loading RETFound MAE (retinal pretrained) from HuggingFace …")
        self.backbone = timm.create_model(
            "hf_hub:bitfount/RETFound_MAE",
            pretrained=pretrained,
            num_classes=0,      # strip head → outputs (B, 1024)
            global_pool="avg",
        )
        self.head = _make_head(self.backbone.embed_dim, num_classes, drop_rate)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_retfound(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    drop_rate: float = 0.3,
    load_pretrained: bool = True,
) -> RETFoundClassifier:
    """Return a RETFound classifier with retinal MAE pretrained weights."""
    model = RETFoundClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        drop_rate=drop_rate,
        pretrained=load_pretrained,
    )
    _print_param_counts(model)
    return model
