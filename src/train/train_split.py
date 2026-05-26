"""Training script for DinoV3_1 glaucoma classifier with a global random split.

All available datasets are concatenated first, then the combined pool is shuffled
once and split into train / val / test partitions.
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import lightning as L
import timm
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    G1020Dataset,
    JRAIGSDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
    RIMONEDataset,
)
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS
from src.models.dino_v3_1 import DinoV3_1


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 896):
    """Return train and evaluation transforms derived from the timm model config."""
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)

    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

    train_tf = transforms.Compose([
        AUGMENTATION_TRANSFORMS,
        timm_train_tf,
    ])

    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _set_source_name(dataset: Dataset, source_name: str) -> Dataset:
    dataset.source_name = source_name  # type: ignore[attr-defined]
    return dataset


def _resolve_dataset_and_index(dataset: Dataset, index: int) -> tuple[Dataset, int]:
    if isinstance(dataset, Subset):
        return _resolve_dataset_and_index(dataset.dataset, dataset.indices[index])

    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        previous_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        return _resolve_dataset_and_index(dataset.datasets[dataset_idx], index - previous_size)

    return dataset, index


def _label_at(dataset: Dataset, index: int) -> int:
    base_dataset, base_index = _resolve_dataset_and_index(dataset, index)

    if isinstance(base_dataset, ACRIMADataset):
        return 1 if "_g_" in base_dataset.image_paths[base_index].name else 0
    if isinstance(base_dataset, FundusTrainValDataset):
        return int(base_dataset.labels[base_index])
    if isinstance(base_dataset, G1020Dataset):
        return int(base_dataset.labels[base_index])
    if isinstance(base_dataset, JRAIGSDataset):
        return int(base_dataset.samples[base_index][1])
    if isinstance(base_dataset, LAGDataset):
        return 1 if base_dataset.image_paths[base_index].name.startswith("g.") else 0
    if isinstance(base_dataset, ORIGADataset):
        return int(base_dataset.samples[base_index][1])
    if isinstance(base_dataset, REFUGE2Dataset):
        return 1 if base_dataset.image_paths[base_index].name.startswith("g") else 0
    if isinstance(base_dataset, RIMONEDataset):
        return 1 if int(base_dataset.labels[base_index]) == 1 else 0

    raise TypeError(f"Unsupported dataset type for label lookup: {type(base_dataset).__name__}")


def _count_glaucoma(dataset: Dataset) -> tuple[int, int]:
    if isinstance(dataset, Subset):
        glaucoma = sum(_label_at(dataset.dataset, index) for index in dataset.indices)
        return glaucoma, len(dataset)

    if isinstance(dataset, ConcatDataset):
        total_glaucoma = 0
        total_samples = 0
        for child in dataset.datasets:
            glaucoma, total = _count_glaucoma(child)
            total_glaucoma += glaucoma
            total_samples += total
        return total_glaucoma, total_samples

    if isinstance(dataset, ACRIMADataset):
        glaucoma = sum(1 for path in dataset.image_paths if "_g_" in path.name)
        return glaucoma, len(dataset)
    if isinstance(dataset, FundusTrainValDataset):
        return sum(dataset.labels), len(dataset)
    if isinstance(dataset, G1020Dataset):
        return sum(dataset.labels), len(dataset)
    if isinstance(dataset, JRAIGSDataset):
        return sum(label for _, label in dataset.samples), len(dataset)
    if isinstance(dataset, LAGDataset):
        glaucoma = sum(1 for path in dataset.image_paths if path.name.startswith("g."))
        return glaucoma, len(dataset)
    if isinstance(dataset, ORIGADataset):
        glaucoma = sum(label for _, label in dataset.samples)
        return glaucoma, len(dataset)
    if isinstance(dataset, REFUGE2Dataset):
        glaucoma = sum(1 for path in dataset.image_paths if path.name.startswith("g"))
        return glaucoma, len(dataset)
    if isinstance(dataset, RIMONEDataset):
        glaucoma = sum(int(label) for label in dataset.labels)
        return glaucoma, len(dataset)
    raise TypeError(f"Unsupported dataset type for counting: {type(dataset).__name__}")


def print_split_info(split_name: str, dataset: Dataset) -> None:
    """Pretty-print size and class balance for a split."""
    rows: dict[str, list[int]] = {}
    total_glaucoma = 0
    total_samples = 0

    if isinstance(dataset, Subset):
        source_dataset = dataset.dataset
        for index in dataset.indices:
            base_dataset, _ = _resolve_dataset_and_index(source_dataset, index)
            dataset_name = getattr(base_dataset, "source_name", type(base_dataset).__name__)
            label = _label_at(source_dataset, index)
            rows.setdefault(dataset_name, [0, 0])
            rows[dataset_name][0] += 1
            rows[dataset_name][1] += label
            total_samples += 1
            total_glaucoma += label
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            dataset_name = getattr(child, "source_name", type(child).__name__)
            glaucoma, total = _count_glaucoma(child)
            rows[dataset_name] = [total, glaucoma]
            total_samples += total
            total_glaucoma += glaucoma
    else:
        dataset_name = getattr(dataset, "source_name", type(dataset).__name__)
        glaucoma, total = _count_glaucoma(dataset)
        rows[dataset_name] = [total, glaucoma]
        total_samples += total
        total_glaucoma += glaucoma

    total_non_glaucoma = total_samples - total_glaucoma
    glaucoma_pct = 100.0 * total_glaucoma / total_samples if total_samples else 0.0

    width = 64
    print(f"\n{'─' * width}")
    print(f"  {split_name}  —  {total_samples} samples total")
    print(f"{'─' * width}")
    print(f"  {'Dataset':<32}  {'Total':>6}  {'Glaucoma':>9}  {'Non-Glauc.':>10}")
    print(f"  {'─' * 60}")
    for dataset_name, (total, glaucoma) in rows.items():
        non_glaucoma = total - glaucoma
        print(f"  {dataset_name:<32}  {total:>6}  {glaucoma:>9}  {non_glaucoma:>10}")
    if len(rows) > 1:
        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':<32}  {total_samples:>6}  {total_glaucoma:>9}  {total_non_glaucoma:>10}")
    print(
        f"  Class balance: {glaucoma_pct:.1f}% glaucoma / "
        f"{100.0 - glaucoma_pct:.1f}% non-glaucoma"
    )
    print(f"{'─' * width}")


def compute_class_weights(train_ds: Dataset) -> list[float]:
    """Return inverse-frequency class weights [w_neg, w_pos] from the training split."""
    glaucoma, total = _count_glaucoma(train_ds)
    non_glaucoma = total - glaucoma
    if non_glaucoma == 0 or glaucoma == 0:
        return [1.0, 1.0]
    w_neg = total / (2.0 * non_glaucoma)
    w_pos = total / (2.0 * glaucoma)
    return [w_neg, w_pos]


class SplitTransformDataset(Dataset):
    """Apply a transform to the image field of a dataset item."""

    def __init__(self, dataset: Dataset, image_transform) -> None:
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            transformed_image = image
        else:
            if hasattr(image, "shape"):
                image = Image.fromarray(image)
            transformed_image = self.image_transform(image)
        sample["image"] = transformed_image
        return sample


def build_combined_dataset(data_dir: str) -> ConcatDataset:
    """Concatenate every available dataset into one global pool."""
    datasets = [
        _set_source_name(FundusTrainValDataset(data_dir=data_dir, split="train", transforms=None), "Fundus(train)"),
        _set_source_name(FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=None), "Fundus(validation)"),
        _set_source_name(JRAIGSDataset(data_dir=data_dir, transforms=None), "JRAIGS(all)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="train", transforms=None), "LAG(train)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="validation", transforms=None), "LAG(validation)"),
        _set_source_name(LAGDataset(data_dir=data_dir, split="test", transforms=None), "LAG(test)"),
        _set_source_name(ORIGADataset(data_dir=data_dir, transforms=None), "ORIGA(all)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="train", partition="hospital", transforms=None), "RIM-ONE(hospital_train)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="test", partition="hospital", transforms=None), "RIM-ONE(hospital_test)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="train", partition="random", transforms=None), "RIM-ONE(random_train)"),
        #_set_source_name(RIMONEDataset(data_dir=data_dir, split="test", partition="random", transforms=None), "RIM-ONE(random_test)"),
    ]
    return ConcatDataset(datasets)


def build_dataloaders(
    data_dir: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 896,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset]:
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)

    combined_ds = build_combined_dataset(data_dir)
    total_samples = len(combined_ds)
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_len = int(total_samples * train_ratio / ratio_sum)
    val_len = int(total_samples * val_ratio / ratio_sum)
    test_len = total_samples - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_raw, val_raw, test_raw = random_split(
        combined_ds,
        [train_len, val_len, test_len],
        generator=generator,
    )

    print_split_info("TRAIN", train_raw)
    print_split_info("VAL  ", val_raw)
    print_split_info("TEST ", test_raw)

    train_ds = SplitTransformDataset(train_raw, train_tf)
    val_ds = SplitTransformDataset(val_raw, eval_tf)
    test_ds = SplitTransformDataset(test_raw, eval_tf)

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

    return train_dl, val_dl, test_dl, train_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DinoV3_1 glaucoma classifier on a global random split"
    )
    parser.add_argument("--data_dir", default="data/datasets")
    parser.add_argument("--backbone", default="vit_huge_plus_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze_backbone_epoch",
        type=int,
        default=3,
        help="If > 0, keep the backbone frozen until this epoch, then unfreeze it.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument(
        "--image_size",
        type=int,
        default=896,
        help="Input image size fed to the backbone. Must be a multiple of 16.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of the combined dataset assigned to training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of the combined dataset assigned to validation.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of the combined dataset assigned to testing.",
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights for [non-glaucoma, glaucoma]. If omitted, weights are computed from the training split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    train_dl, val_dl, test_dl, train_raw = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    class_weights = args.class_weights if args.class_weights is not None else compute_class_weights(train_raw)
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

    logger = CSVLogger(save_dir="lightning_logs", name="dinov3_1_split")
    version_number = logger.version
    print(f"Logging to: lightning_logs/dinov3_1_split/version_{version_number}")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/version_{version_number}",
            filename=f"dinov3_1_split_v{version_number}-" + "{epoch:02d}-{val_auc:.4f}",
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
        deterministic=False,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print("Testing on the held-out split")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()