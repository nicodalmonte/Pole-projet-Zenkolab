"""Training script for DinoV3_1 glaucoma classifier.

Dataset split strategy
----------------------
Train  : ACRIMA (all) + Fundus (train) + LAG (train) + ORIGA (all)
Val    : Fundus (validation) + LAG (validation)
Test   : REFUGE2 (train + val + test, held-out, evaluated after training)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm 
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
    JRAIGSDataset,
)
from src.models.dino_v3_1 import DinoV3_1
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 896):
    """Return (train_transform, eval_transform) derived from the timm model config.
    
    Training transforms include data augmentations (rotations, affine transforms, etc.)
    defined in src.augmentations.AUGMENTATION_TRANSFORMS.
    
    Evaluation transforms do not include augmentations.
    """
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    # Override the crop size so the full image_size is passed to the backbone.
    # DINOv3 uses patch_size=16, so any multiple of 16 is valid.
    data_cfg["input_size"] = (3, image_size, image_size)
    
    # Build the timm transforms
    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)
    
    # Compose timm transforms with augmentations for training
    train_tf = transforms.Compose([
        AUGMENTATION_TRANSFORMS,
        timm_train_tf,
    ])
    
    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _count_glaucoma(ds) -> tuple[int, int]:
    """Return (n_glaucoma, total) by inspecting stored paths/labels.

    No images are opened — only filename metadata / label lists are read.
    """
    if isinstance(ds, Subset):
        base_ds = ds.dataset
        if hasattr(base_ds, "samples"):
            g = sum(base_ds.samples[i][1] for i in ds.indices)
            return g, len(ds)
        return 0, len(ds)

    if isinstance(ds, ACRIMADataset):
        g = sum(1 for p in ds.image_paths if "_g_" in p.name)
    elif isinstance(ds, LAGDataset):
        g = sum(1 for p in ds.image_paths if p.name.startswith("g."))
    elif isinstance(ds, ORIGADataset):
        g = sum(lbl for _, lbl in ds.samples)
    elif isinstance(ds, REFUGE2Dataset):
        g = sum(1 for p in ds.image_paths if p.name.startswith("g"))
    elif isinstance(ds, FundusTrainValDataset):
        g = sum(ds.labels)
    elif isinstance(ds, JRAIGSDataset):
        g = sum(lbl for _, lbl in ds.samples)
    else:
        return 0, len(ds)
    return g, len(ds)


def print_split_info(split_name: str, ds) -> None:
    """Pretty-print size and class balance for a split (ConcatDataset or Dataset)."""
    sub_datasets = ds.datasets if hasattr(ds, "datasets") else [ds]
    rows: list[tuple[str, int, int, int]] = []
    total_g = total_ng = 0
    for sub in sub_datasets:
        g, tot = _count_glaucoma(sub)
        ng = tot - g
        total_g += g
        total_ng += ng
        rows.append((type(sub).__name__, tot, g, ng))
    total = total_g + total_ng
    glaucoma_pct = 100.0 * total_g / total if total else 0.0

    W = 64
    print(f"\n{'─' * W}")
    print(f"  {split_name}  —  {total} samples total")
    print(f"{'─' * W}")
    print(f"  {'Dataset':<32}  {'Total':>6}  {'Glaucoma':>9}  {'Non-Glauc.':>10}")
    print(f"  {'─' * 60}")
    for ds_name, tot, g, ng in rows:
        print(f"  {ds_name:<32}  {tot:>6}  {g:>9}  {ng:>10}")
    if len(rows) > 1:
        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':<32}  {total:>6}  {total_g:>9}  {total_ng:>10}")
    print(
        f"  Class balance: {glaucoma_pct:.1f}% glaucoma / "
        f"{100.0 - glaucoma_pct:.1f}% non-glaucoma"
    )
    print(f"{'─' * W}")

def compute_class_weights(train_ds) -> list[float]:
    """Return inverse-frequency class weights [w_neg, w_pos] from the training ConcatDataset."""
    sub_datasets = train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds]
    total_g = total_ng = 0
    for sub in sub_datasets:
        g, tot = _count_glaucoma(sub)
        total_g += g
        total_ng += tot - g
    total = total_g + total_ng
    if total_ng == 0 or total_g == 0:
        return [1.0, 1.0]
    w_neg = total / (2.0 * total_ng)
    w_pos = total / (2.0 * total_g)
    return [w_neg, w_pos]


def build_dataloaders(
    data_dir: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 896,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)
    target_train_size = 8_000

    ## --- Train ---
    #train_ds = ConcatDataset([
    #    ACRIMADataset(data_dir=data_dir, split="train", transforms=train_tf),
    #    LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
    #    ORIGADataset(data_dir=data_dir, split="train", transforms=train_tf),
    #])
#
    ## --- Val ---
    #val_ds = ConcatDataset([
    #    FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
    #    LAGDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
    #])
#
    ## --- Test (REFUGE2, all three splits) ---
    #test_ds = ConcatDataset([
    #    REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),
    #])

    jraigs_train = JRAIGSDataset(data_dir=data_dir, transforms=train_tf)
    glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 1]
    non_glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 0]

    remaining_slots = max(target_train_size - len(glaucoma_indices), 0)
    if remaining_slots >= len(non_glaucoma_indices):
        selected_non_glaucoma = non_glaucoma_indices
    else:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(non_glaucoma_indices), generator=g)[:remaining_slots].tolist()
        selected_non_glaucoma = [non_glaucoma_indices[i] for i in perm]

    selected_indices = glaucoma_indices + selected_non_glaucoma
    train_ds = ConcatDataset([
        Subset(jraigs_train, selected_indices),
    ])
    #traind_ds = jraigs_train ## if I want all to use all the images
    

    val_ds = ConcatDataset([
        ACRIMADataset(data_dir=data_dir, split="train", transforms=eval_tf),
        ORIGADataset(data_dir=data_dir, split="train", transforms=eval_tf),
        LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf)
    ])

    test_ds = ConcatDataset([
        REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),        
    ])

    # -- Print dataset details before building DataLoaders --
    print_split_info("TRAIN", train_ds)
    print_split_info("VAL  ", val_ds)
    print_split_info("TEST (REFUGE2)", test_ds)

    pin = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)} (REFUGE2)")

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DinoV3_1 glaucoma classifier")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--backbone", default="vit_huge_plus_patch16_dinov3.lvd1689m")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--unfreeze_backbone_epoch",
        type=int,
        default=3,
        help="If > 0, keep the backbone frozen until this epoch, then unfreeze it.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument(
        "--image_size",
        type=int,
        default=896,
        help="Input image size fed to the backbone. Must be a multiple of 16 "
             "(patch size, DINOv3). Default 896 = 4×224, i.e. 56×56 patches — "
             "chosen to preserve detail from high-res datasets (ORIGA ~3072px, "
             "REFUGE2 ~2000px) while still being a clean 4× multiple of the "
             "pretrained resolution (minimal positional-embedding interpolation "
             "artefact). Lower-resolution datasets (ACRIMA ~332px median, LAG "
             "500px) are upscaled by the transform. Reduce to 448 or 336 if "
             "VRAM is tight.",
    )
    p.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights for [non-glaucoma, glaucoma]. "
             "If omitted, weights are computed automatically from the training "
             "set using inverse-frequency balancing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    class_weights = args.class_weights if args.class_weights is not None else compute_class_weights(train_dl.dataset)
    print(f"\nClass weights — non-glaucoma: {class_weights[0]:.4f}  glaucoma: {class_weights[1]:.4f}")

    model = DinoV3_1(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.image_size,
        class_weights=class_weights,
        unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
    )

    logger = CSVLogger(save_dir="lightning_logs", name="dinov3_1")
    version_number = logger.version
    print(f"L Logging to: lightning_logs/dinov3_1/version_{version_number}")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/version_{version_number}",
            filename=f"dinov3_1_v{version_number}-"+"{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            min_delta=1e-3,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=3),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision=args.precision,
        devices=args.devices,
        log_every_n_steps=10,
        deterministic=False,  # faster with non-deterministic ops
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,  # handled by RichModelSummary
    )
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)

    # --- Test on REFUGE2 using the best checkpoint ---
    print("\n" + "=" * 60)
    print("Testing on REFUGE2 (held-out test set)")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
