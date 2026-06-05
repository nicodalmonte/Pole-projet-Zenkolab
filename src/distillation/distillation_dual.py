"""Training script for dual-teacher distilled student glaucoma classifier.

Teachers  : DinoV3_1 (DINO) + EvaViT (EVA)
Student   : any timm backbone (default: DINO small, 224 px)

Dataset split strategy
----------------------
Train  : JRAIGS (all glaucoma + up to 8 000 non-glaucoma, seed 42)
Val    : ACRIMA (train) + ORIGA (train) + LAG (train)
Test   : REFUGE2 (train, held-out)

Image sizes during training
----------------------------
image_a  (student + DINO teacher) : --image_size      (default 224 px)
image_b  (EVA teacher only)       : --eva_image_size   (default 448 px)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from torchvision import transforms

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    JRAIGSDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
)
from src.models.dino_v3_1 import DinoV3_1
from src.models.eva_vit import EvaViT
from src.models.student_dual import StudentGlaucomaDualDistilled
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS


# ---------------------------------------------------------------------------
# Dual-transform dataset wrapper
# ---------------------------------------------------------------------------

class DualImageDataset(Dataset):
    """Wraps a base dataset (instantiated with transforms=None) to return two
    transformed versions of each image — one per teacher resolution.

    The base dataset __getitem__ must return {"image": np.ndarray, "label": tensor, ...}.
    Output keys are "image_a" (student/DINO), "image_b" (EVA), and "label".
    """

    def __init__(self, base_dataset: Dataset, transform_a, transform_b) -> None:
        # Base dataset with no transform applied (returns raw numpy H×W×C arrays)
        self.base_dataset = base_dataset
        # transform_a produces student/DINO-resolution tensors (e.g. 224 px)
        self.transform_a = transform_a
        # transform_b produces EVA-resolution tensors (e.g. 448 px)
        self.transform_b = transform_b

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        # Retrieve raw item; "image" is a numpy uint8 array (H, W, 3)
        item = self.base_dataset[idx]
        # Convert numpy array back to PIL Image so torchvision transforms can run
        pil_image = Image.fromarray(item["image"])
        return {
            # Student/DINO view at the student training resolution
            "image_a": self.transform_a(pil_image),
            # EVA view at EVA's training resolution
            "image_b": self.transform_b(pil_image),
            # Ground-truth label is shared between both views
            "label": item["label"],
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_student_transforms(backbone_name: str, image_size: int):
    """Return (train_transform, eval_transform) for the student backbone."""
    # Resolve timm's default preprocessing config for the given backbone
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    # Override spatial resolution to match the student's training size
    data_cfg["input_size"] = (3, image_size, image_size)

    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

    # Prepend ophthalmology-specific augmentations before the timm pipeline
    train_tf = transforms.Compose([AUGMENTATION_TRANSFORMS, timm_train_tf])

    return train_tf, timm_eval_tf


def build_eva_transform(backbone_name: str, image_size: int):
    """Return the eval-only transform for EVA teacher inputs (no augmentation on teacher images)."""
    # Resolve EVA's default preprocessing config
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    # Override spatial resolution to EVA's training resolution
    data_cfg["input_size"] = (3, image_size, image_size)
    # No random augmentation: teacher receives a clean, deterministic crop
    return timm.data.create_transform(**data_cfg, is_training=False)


# ---------------------------------------------------------------------------
# Data helpers (adapted from distilation.py to handle DualImageDataset)
# ---------------------------------------------------------------------------

def _count_glaucoma(ds) -> tuple[int, int]:
    """Return (n_glaucoma, total) by inspecting stored paths/labels without loading images."""
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


def _unwrap(ds) -> object:
    """Unwrap DualImageDataset to reach the underlying base dataset for counting."""
    # DualImageDataset stores the real dataset in .base_dataset
    return ds.base_dataset if isinstance(ds, DualImageDataset) else ds


def print_split_info(split_name: str, ds) -> None:
    """Pretty-print size and class balance for a split."""
    # Handle both ConcatDataset (has .datasets) and a single dataset
    sub_datasets = ds.datasets if hasattr(ds, "datasets") else [ds]
    rows: list[tuple[str, int, int, int]] = []
    total_g = total_ng = 0

    for sub in sub_datasets:
        # Unwrap DualImageDataset before counting to reach the real dataset
        real = _unwrap(sub)
        g, tot = _count_glaucoma(real)
        ng = tot - g
        total_g += g
        total_ng += ng
        rows.append((type(real).__name__, tot, g, ng))

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
    """Return inverse-frequency class weights [w_neg, w_pos] from the training split."""
    sub_datasets = train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds]
    total_g = total_ng = 0

    for sub in sub_datasets:
        # Unwrap DualImageDataset before counting
        real = _unwrap(sub)
        g, tot = _count_glaucoma(real)
        total_g += g
        total_ng += tot - g

    total = total_g + total_ng
    if total_ng == 0 or total_g == 0:
        return [1.0, 1.0]

    w_neg = total / (2.0 * total_ng)
    w_pos = total / (2.0 * total_g)
    return [w_neg, w_pos]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    student_backbone: str,
    eva_backbone: str,
    batch_size: int,
    num_workers: int,
    student_image_size: int = 224,
    eva_image_size: int = 448,
    dataset: str = "jraigs",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    student_train_tf, student_eval_tf = build_student_transforms(student_backbone, student_image_size)
    eva_tf = build_eva_transform(eva_backbone, eva_image_size)

    if dataset == "lag_origa":
        lag_raw = LAGDataset(data_dir=data_dir, split="train", transforms=None)
        train_ds = DualImageDataset(lag_raw, student_train_tf, eva_tf)
        val_ds = ConcatDataset([LAGDataset(data_dir=data_dir, split="validation", transforms=student_eval_tf)])
        test_ds = ConcatDataset([ORIGADataset(data_dir=data_dir, transforms=student_eval_tf)])
    else:  # jraigs (default)
        target_train_size = 8_000

        jraigs_raw = JRAIGSDataset(data_dir=data_dir, transforms=None)
        glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_raw.samples) if lbl == 1]
        non_glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_raw.samples) if lbl == 0]

        remaining_slots = max(target_train_size - len(glaucoma_indices), 0)
        if remaining_slots >= len(non_glaucoma_indices):
            selected_non_glaucoma = non_glaucoma_indices
        else:
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(non_glaucoma_indices), generator=g)[:remaining_slots].tolist()
            selected_non_glaucoma = [non_glaucoma_indices[i] for i in perm]

        jraigs_subset = Subset(jraigs_raw, glaucoma_indices + selected_non_glaucoma)
        train_ds = DualImageDataset(jraigs_subset, student_train_tf, eva_tf)

        val_ds = ConcatDataset([
            ACRIMADataset(data_dir=data_dir, split="train", transforms=student_eval_tf),
            ORIGADataset(data_dir=data_dir, split="train", transforms=student_eval_tf),
            LAGDataset(data_dir=data_dir, split="train", transforms=student_eval_tf),
        ])
        test_ds = ConcatDataset([REFUGE2Dataset(data_dir=data_dir, split="train", transforms=student_eval_tf)])

    print_split_info("TRAIN", train_ds)
    print_split_info("VAL  ", val_ds)
    print_split_info("TEST ", test_ds)

    pin = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-teacher distillation: DINO + EVA → student")

    p.add_argument("--data_dir", default="data/datasets")

    # Teacher checkpoints
    p.add_argument("--dino_ckpt", required=True, help="Path to trained DinoV3_1 checkpoint.")
    p.add_argument("--eva_ckpt", required=True, help="Path to trained EvaViT checkpoint.")

    # Student architecture
    p.add_argument("--student_backbone", default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--student_pretrained", action="store_true", default=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    # Student/DINO teacher resolution
    p.add_argument("--image_size", type=int, default=224)
    # EVA teacher resolution — loaded from EVA hparams if not specified
    p.add_argument("--eva_image_size", type=int, default=448)

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=16)

    # Distillation loss weights
    p.add_argument("--kd_lambda_dino", type=float, default=0.4, help="λ_dino: weight on DINO KD loss.")
    p.add_argument("--kd_lambda_eva", type=float, default=0.4, help="λ_eva: weight on EVA KD loss.")
    # Hard CE loss weight is implicitly (1 - λ_dino - λ_eva)
    p.add_argument("--kd_temperature", type=float, default=4.0, help="Shared KD temperature T.")

    # Misc
    p.add_argument("--checkpoint_dir", default="checkpoints/student_dual")
    p.add_argument("--resume", default=None, help="Path to student checkpoint to resume from.")
    p.add_argument("--devices", default="auto")
    p.add_argument("--dataset", default="jraigs", choices=["jraigs", "lag_origa"],
                   help="Dataset config: 'jraigs' (default) or 'lag_origa' (LAG train, ORIGA test).")
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights. If omitted, computed from training split.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    # Load DINO teacher from checkpoint (restores backbone_name and all hparams)
    print("Loading DINO teacher...")
    dino_teacher = DinoV3_1.load_from_checkpoint(args.dino_ckpt)
    dino_teacher.eval()
    for p in dino_teacher.parameters():
        p.requires_grad = False

    # Load EVA teacher from checkpoint (restores backbone_name and img_size hparams)
    print("Loading EVA teacher...")
    eva_teacher = EvaViT.load_from_checkpoint(args.eva_ckpt)
    eva_teacher.eval()
    for p in eva_teacher.parameters():
        p.requires_grad = False

    # Read EVA backbone name from checkpoint hparams to build the correct transform
    eva_backbone_name = eva_teacher.hparams.backbone_name

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        student_backbone=args.student_backbone,
        eva_backbone=eva_backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        student_image_size=args.image_size,
        eva_image_size=args.eva_image_size,
        dataset=args.dataset,
    )

    class_weights = (
        args.class_weights
        if args.class_weights is not None
        else compute_class_weights(train_dl.dataset)
    )
    print(
        f"\nClass weights — non-glaucoma: {class_weights[0]:.4f}  "
        f"glaucoma: {class_weights[1]:.4f}"
    )
    # Log the effective hard-loss weight for transparency
    hard_weight = 1.0 - args.kd_lambda_dino - args.kd_lambda_eva
    print(
        f"Loss weights  — λ_dino: {args.kd_lambda_dino}  "
        f"λ_eva: {args.kd_lambda_eva}  hard: {hard_weight:.2f}"
    )

    # Build the student with both frozen teachers attached
    student = StudentGlaucomaDualDistilled(
        backbone_name=args.student_backbone,
        pretrained=args.student_pretrained,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.image_size,
        class_weights=class_weights,
        teacher_dino=dino_teacher,
        teacher_eva=eva_teacher,
        kd_lambda_dino=args.kd_lambda_dino,
        kd_lambda_eva=args.kd_lambda_eva,
        kd_temperature=args.kd_temperature,
        freeze_backbone_epochs=0,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            # Include both lambdas in the filename for easy identification
            filename="student_dual-{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=10, min_delta=1e-3, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=3),
    ]

    logger = CSVLogger(save_dir="lightning_logs", name="student_dual_distilled")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision=args.precision,
        devices=args.devices,
        log_every_n_steps=10,
        deterministic=False,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    trainer.fit(student, train_dl, val_dl, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print("Testing dual-distilled student on held-out test set")
    print("=" * 60)
    trainer.test(student, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
