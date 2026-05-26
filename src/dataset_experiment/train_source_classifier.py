"""Train a CNN to predict the source dataset of each fundus image.

This experiment quantifies dataset bias:
if a small CNN can easily identify where an image comes from, then
source-specific artifacts are likely strong.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from PIL import Image
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision import transforms

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")


def _proportional_split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must have a positive sum.")

    raw = [total * r / ratio_sum for r in ratios]
    counts = [int(v) for v in raw]
    remainder = total - sum(counts)

    fractional_order = sorted(
        range(len(raw)),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    for i in fractional_order[:remainder]:
        counts[i] += 1

    return counts[0], counts[1], counts[2]


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------


class SourceLabeledDataset(Dataset):
    """Wrap a dataset and replace the target with a source-id label."""

    def __init__(self, dataset: Dataset, source_name: str, source_idx: int) -> None:
        self.dataset = dataset
        self.source_name = source_name
        self.source_idx = source_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = dict(self.dataset[index])
        sample["glaucoma_label"] = sample.get("label")
        sample["label"] = torch.tensor(self.source_idx, dtype=torch.long)
        sample["source_name"] = self.source_name
        return sample


class TransformingSubset(Dataset):
    """Apply an image transform to a subset sample dictionary."""

    def __init__(self, dataset: Dataset, image_transform) -> None:
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = dict(self.dataset[index])
        image = sample["image"]

        if isinstance(image, torch.Tensor):
            transformed = image
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            transformed = self.image_transform(image)

        sample["image"] = transformed
        return sample


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


class DatasetSourceDataModule(L.LightningDataModule):
    """Build a stratified train/val/test split for source classification."""

    def __init__(
        self,
        data_dir: str = "data/datasets",
        image_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 8,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        include_rim_one: bool = True,
        selected_sources: list[str] | None = None,
        truncate_largest_source: bool = True,
        largest_source_max_ratio: float = 2.0,
        largest_source_hard_cap: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.include_rim_one = include_rim_one
        self.selected_sources = {s.lower() for s in selected_sources} if selected_sources else None
        self.truncate_largest_source = truncate_largest_source
        self.largest_source_max_ratio = largest_source_max_ratio
        self.largest_source_hard_cap = largest_source_hard_cap

        self.class_names: list[str] = []
        self.class_weights: torch.Tensor | None = None
        self.skipped_notes: list[str] = []
        self.balance_notes: list[str] = []
        self.split_summary: dict[str, dict[str, int]] = {}

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._setup_done = False

    def _build_transforms(self):
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        train_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        return train_tf, eval_tf

    def _source_specs(self) -> list[tuple[str, list[Callable[[], Dataset]]]]:
        specs: list[tuple[str, list[Callable[[], Dataset]]]] = [
            (
                "ACRIMA",
                [
                    lambda: ACRIMADataset(data_dir=self.data_dir, split="train", transforms=None),
                ],
            ),
            (
                "Fundus_Train_Val_Data",
                [
                    lambda: FundusTrainValDataset(data_dir=self.data_dir, split="train", transforms=None),
                    lambda: FundusTrainValDataset(data_dir=self.data_dir, split="validation", transforms=None),
                ],
            ),
            (
                "G1020",
                [
                    lambda: G1020Dataset(data_dir=self.data_dir, split="train", transforms=None),
                    lambda: G1020Dataset(data_dir=self.data_dir, split="test", transforms=None),
                ],
            ),
            (
                "JRAIGS",
                [
                    lambda: JRAIGSDataset(data_dir=self.data_dir, split="train", transforms=None),
                ],
            ),
            (
                "LAG",
                [
                    lambda: LAGDataset(data_dir=self.data_dir, split="train", transforms=None),
                    lambda: LAGDataset(data_dir=self.data_dir, split="validation", transforms=None),
                    lambda: LAGDataset(data_dir=self.data_dir, split="test", transforms=None),
                ],
            ),
            (
                "ORIGA",
                [
                    lambda: ORIGADataset(data_dir=self.data_dir, split="train", transforms=None),
                ],
            ),
            (
                "REFUGE2",
                [
                    lambda: REFUGE2Dataset(data_dir=self.data_dir, split="train", transforms=None),
                    lambda: REFUGE2Dataset(data_dir=self.data_dir, split="val", transforms=None),
                    lambda: REFUGE2Dataset(data_dir=self.data_dir, split="test", transforms=None),
                ],
            ),
        ]

        if self.include_rim_one:
            specs.append(
                (
                    "RIM_ONE",
                    [
                        lambda: RIMONEDataset(
                            data_dir=self.data_dir,
                            split="train",
                            partition="hospital",
                            transforms=None,
                        ),
                        lambda: RIMONEDataset(
                            data_dir=self.data_dir,
                            split="test",
                            partition="hospital",
                            transforms=None,
                        ),
                    ],
                )
            )

        return specs

    def _is_selected(self, source_name: str) -> bool:
        if self.selected_sources is None:
            return True
        return source_name.lower() in self.selected_sources

    def _maybe_truncate_largest_source(self, wrapped: list[Dataset]) -> list[Dataset]:
        if not self.truncate_largest_source or len(wrapped) < 2:
            return wrapped

        counts = [len(ds) for ds in wrapped]
        ranked = sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True)
        largest_idx = ranked[0]
        second_idx = ranked[1]

        largest_count = counts[largest_idx]
        second_count = counts[second_idx]
        source_name = self.class_names[largest_idx]
        second_name = self.class_names[second_idx]

        ratio_cap = max(1, int(second_count * self.largest_source_max_ratio))
        target_count = ratio_cap
        if self.largest_source_hard_cap is not None:
            target_count = min(target_count, self.largest_source_hard_cap)

        if target_count >= largest_count:
            note = (
                f"[BALANCE] No truncation: largest source {source_name} has {largest_count} samples; "
                f"cap is {target_count} (2nd largest: {second_name}={second_count}, "
                f"ratio={self.largest_source_max_ratio:.2f})."
            )
            print(note)
            self.balance_notes.append(note)
            return wrapped

        generator = torch.Generator().manual_seed(self.seed)
        keep_indices = torch.randperm(largest_count, generator=generator)[:target_count].tolist()
        wrapped[largest_idx] = Subset(wrapped[largest_idx], keep_indices)

        note = (
            f"[BALANCE] Truncated largest source {source_name}: {largest_count} -> {target_count} samples "
            f"(2nd largest: {second_name}={second_count}, ratio cap={self.largest_source_max_ratio:.2f})."
        )
        print(note)
        self.balance_notes.append(note)
        return wrapped

    def _build_source_pools(self) -> tuple[ConcatDataset, list[int]]:
        wrapped: list[Dataset] = []

        self.class_names = []
        self.skipped_notes = []
        self.balance_notes = []

        for source_name, builders in self._source_specs():
            if not self._is_selected(source_name):
                note = f"[SKIP] Source {source_name} was filtered out by --sources."
                print(note)
                self.skipped_notes.append(note)
                continue

            parts: list[Dataset] = []
            part_errors: list[str] = []

            for build_fn in builders:
                try:
                    dataset = build_fn()
                except Exception as exc:  # noqa: BLE001
                    part_errors.append(str(exc))
                    continue
                parts.append(dataset)

            if not parts:
                note = f"[WARN] Source {source_name} skipped (all splits failed to load)."
                print(note)
                self.skipped_notes.append(note)
                for err in part_errors:
                    err_note = f"       -> {err}"
                    print(err_note)
                    self.skipped_notes.append(err_note)
                continue

            pooled = parts[0] if len(parts) == 1 else ConcatDataset(parts)
            source_idx = len(self.class_names)
            self.class_names.append(source_name)

            source_wrapped = SourceLabeledDataset(
                dataset=pooled,
                source_name=source_name,
                source_idx=source_idx,
            )
            wrapped.append(source_wrapped)

            print(f"[LOAD] {source_name:<22} -> {len(source_wrapped):>6} images")

        if len(wrapped) < 2:
            raise RuntimeError(
                "Need at least 2 loaded sources to run source classification. "
                "Check dataset paths and optional --sources filter."
            )

        wrapped = self._maybe_truncate_largest_source(wrapped)

        labels: list[int] = []
        for source_idx, source_dataset in enumerate(wrapped):
            labels.extend([source_idx] * len(source_dataset))

        print(f"[INFO] Loaded {len(self.class_names)} source classes.")
        return ConcatDataset(wrapped), labels

    def _stratified_split_indices(self, labels: list[int]) -> tuple[list[int], list[int], list[int]]:
        class_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            class_to_indices.setdefault(label, []).append(idx)

        generator = torch.Generator().manual_seed(self.seed)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for class_idx, class_indices in sorted(class_to_indices.items()):
            index_tensor = torch.tensor(class_indices, dtype=torch.long)
            perm = torch.randperm(len(index_tensor), generator=generator)
            shuffled = index_tensor[perm].tolist()

            n_train, n_val, n_test = _proportional_split_counts(
                total=len(shuffled),
                ratios=(self.train_ratio, self.val_ratio, self.test_ratio),
            )

            train_indices.extend(shuffled[:n_train])
            val_indices.extend(shuffled[n_train : n_train + n_val])
            test_indices.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

            source_name = self.class_names[class_idx]
            print(
                f"[SPLIT] {source_name:<22} total={len(shuffled):>6} "
                f"train={n_train:>6} val={n_val:>6} test={n_test:>6}"
            )

        def _shuffle_in_place(indices: list[int]) -> list[int]:
            if not indices:
                return indices
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            perm = torch.randperm(len(idx_tensor), generator=generator)
            return idx_tensor[perm].tolist()

        return (
            _shuffle_in_place(train_indices),
            _shuffle_in_place(val_indices),
            _shuffle_in_place(test_indices),
        )

    def _split_counts(self, labels: list[int], indices: list[int]) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for idx in indices:
            source_idx = labels[idx]
            counts[self.class_names[source_idx]] += 1
        return counts

    def _print_split_summary(self, labels: list[int], train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> None:
        summaries = {
            "full": self._split_counts(labels, list(range(len(labels)))),
            "train": self._split_counts(labels, train_idx),
            "val": self._split_counts(labels, val_idx),
            "test": self._split_counts(labels, test_idx),
        }
        self.split_summary = summaries

        print("\n" + "=" * 86)
        print("Source distribution per split")
        print("=" * 86)

        for split_name in ("full", "train", "val", "test"):
            split_counts = summaries[split_name]
            split_total = sum(split_counts.values())
            print(f"\n[{split_name.upper()}] total={split_total}")
            for source_name in self.class_names:
                count = split_counts[source_name]
                pct = (100.0 * count / split_total) if split_total else 0.0
                print(f"  - {source_name:<22} count={count:>6}  pct={pct:>6.2f}%")

        print("\n" + "=" * 86)

    def _compute_class_weights(self, labels: list[int], train_indices: list[int]) -> torch.Tensor:
        train_counts = torch.zeros(len(self.class_names), dtype=torch.float32)
        for idx in train_indices:
            train_counts[labels[idx]] += 1.0

        total = train_counts.sum()
        if total == 0:
            return torch.ones(len(self.class_names), dtype=torch.float32)

        weights = torch.zeros_like(train_counts)
        non_zero = train_counts > 0
        weights[non_zero] = total / (len(self.class_names) * train_counts[non_zero])
        weights[~non_zero] = 0.0
        return weights

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return

        train_tf, eval_tf = self._build_transforms()
        combined_dataset, labels = self._build_source_pools()
        train_idx, val_idx, test_idx = self._stratified_split_indices(labels)

        if not train_idx or not val_idx or not test_idx:
            raise RuntimeError(
                "One split is empty after stratified split. "
                "Adjust train/val/test ratios or source selection."
            )

        train_raw = Subset(combined_dataset, train_idx)
        val_raw = Subset(combined_dataset, val_idx)
        test_raw = Subset(combined_dataset, test_idx)

        self.class_weights = self._compute_class_weights(labels, train_idx)
        self._print_split_summary(labels, train_idx, val_idx, test_idx)

        print("Class weights (inverse frequency on train split):")
        for class_name, weight in zip(self.class_names, self.class_weights.tolist(), strict=True):
            print(f"  - {class_name:<22} weight={weight:.4f}")

        self._train_dataset = TransformingSubset(train_raw, train_tf)
        self._val_dataset = TransformingSubset(val_raw, eval_tf)
        self._test_dataset = TransformingSubset(test_raw, eval_tf)
        self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")

        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")

        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def export_split_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        split_json = {
            "class_names": self.class_names,
            "split_summary": self.split_summary,
            "class_weights": self.class_weights.tolist() if self.class_weights is not None else None,
            "skipped_notes": self.skipped_notes,
            "balance_notes": self.balance_notes,
        }
        with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
            json.dump(split_json, handle, indent=2)

        with open(output_dir / "split_distribution.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "source", "count", "fraction"])
            for split_name, counts in self.split_summary.items():
                total = sum(counts.values())
                for source_name in self.class_names:
                    count = counts[source_name]
                    fraction = (count / total) if total else 0.0
                    writer.writerow([split_name, source_name, count, f"{fraction:.6f}"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SourceDatasetCNN(L.LightningModule):
    """Simple CNN for multiclass source prediction."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.class_names = class_names

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.register_buffer("loss_weight", class_weights if class_weights is not None else None)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

        self.has_topk = num_classes >= 3
        if self.has_topk:
            top_k = min(3, num_classes)
            self.val_topk = MulticlassAccuracy(num_classes=num_classes, average="micro", top_k=top_k)
            self.test_topk = MulticlassAccuracy(num_classes=num_classes, average="micro", top_k=top_k)

        self._test_rows: list[tuple[str, int, int, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.loss_weight)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["label"]
        batch_size = targets.size(0)

        logits = self(images)
        loss = self._loss(logits, targets)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        targets = batch["label"]
        batch_size = targets.size(0)

        logits = self(images)
        loss = self._loss(logits, targets)
        preds = logits.argmax(dim=1)

        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_cm.update(preds, targets)

        if self.has_topk:
            self.val_topk.update(logits, targets)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()

        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)
        self.log("val_f1_macro", val_f1, prog_bar=True, sync_dist=True)
        self.log("val_precision_macro", val_precision, sync_dist=True)
        self.log("val_recall_macro", val_recall, sync_dist=True)
        self.log("val_balanced_acc", val_recall, prog_bar=True, sync_dist=True)

        if self.has_topk:
            self.log("val_top3_acc", self.val_topk.compute(), sync_dist=True)

        cm = self.val_cm.compute()
        self._log_per_class_metrics(cm, split="val")
        self._save_confusion_matrix(cm, split="val", epoch=self.current_epoch)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_cm.reset()
        if self.has_topk:
            self.val_topk.reset()

    def on_test_epoch_start(self) -> None:
        self._test_rows.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        targets = batch["label"]
        batch_size = targets.size(0)

        logits = self(images)
        loss = self._loss(logits, targets)

        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)

        self.test_acc.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_cm.update(preds, targets)

        if self.has_topk:
            self.test_topk.update(logits, targets)

        paths = batch.get("path", [""] * len(preds))
        for path, true_label, pred_label, conf in zip(
            paths,
            targets.detach().cpu().tolist(),
            preds.detach().cpu().tolist(),
            confidences.detach().cpu().tolist(),
            strict=True,
        ):
            self._test_rows.append((str(path), true_label, pred_label, conf))

        self.log("test_loss", loss, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def on_test_epoch_end(self) -> None:
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()

        self.log("test_acc", test_acc, prog_bar=True, sync_dist=True)
        self.log("test_f1_macro", test_f1, prog_bar=True, sync_dist=True)
        self.log("test_precision_macro", test_precision, sync_dist=True)
        self.log("test_recall_macro", test_recall, sync_dist=True)
        self.log("test_balanced_acc", test_recall, prog_bar=True, sync_dist=True)

        if self.has_topk:
            self.log("test_top3_acc", self.test_topk.compute(), sync_dist=True)

        cm = self.test_cm.compute()
        self._log_per_class_metrics(cm, split="test")
        self._save_confusion_matrix(cm, split="test", epoch=self.current_epoch)
        self._save_test_predictions()

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_cm.reset()
        if self.has_topk:
            self.test_topk.reset()

    def _log_per_class_metrics(self, cm: torch.Tensor, split: str) -> None:
        cm_float = cm.float()
        tp = cm_float.diag()
        support = cm_float.sum(dim=1).clamp_min(1.0)
        pred_support = cm_float.sum(dim=0).clamp_min(1.0)

        recalls = tp / support
        precisions = tp / pred_support

        for idx, source_name in enumerate(self.class_names):
            suffix = _safe_name(source_name)
            self.log(f"{split}_recall_{suffix}", recalls[idx], sync_dist=False)
            self.log(f"{split}_precision_{suffix}", precisions[idx], sync_dist=False)

    def _logger_dir(self) -> Path | None:
        if self.logger is None:
            return None
        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return None
        return Path(log_dir)

    def _save_confusion_matrix(self, cm: torch.Tensor, split: str, epoch: int) -> None:
        if not self.trainer.is_global_zero:
            return

        log_dir = self._logger_dir()
        if log_dir is None:
            return

        output_dir = log_dir / "confusion_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{split}_epoch_{epoch:03d}.csv"

        with open(out_file, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\pred", *self.class_names])
            for row_idx, source_name in enumerate(self.class_names):
                row_values = cm[row_idx].detach().cpu().tolist()
                writer.writerow([source_name, *row_values])

    def _save_test_predictions(self) -> None:
        if not self.trainer.is_global_zero:
            return

        log_dir = self._logger_dir()
        if log_dir is None:
            return

        out_file = log_dir / "test_predictions.csv"
        with open(out_file, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "true_label", "true_source", "pred_label", "pred_source", "confidence"])
            for path, true_label, pred_label, confidence in self._test_rows:
                writer.writerow(
                    [
                        path,
                        true_label,
                        self.class_names[true_label],
                        pred_label,
                        self.class_names[pred_label],
                        f"{confidence:.6f}",
                    ]
                )

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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_macro",
                "interval": "epoch",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify from which dataset each image comes from."
    )
    parser.add_argument("--data_dir", default="data/datasets")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source names to include (case-insensitive).",
    )
    parser.add_argument(
        "--exclude_rim_one",
        action="store_true",
        help="Exclude RIM-ONE from the source pool.",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--disable_largest_truncation",
        action="store_true",
        help="Disable automatic truncation of the largest source dataset.",
    )
    parser.add_argument(
        "--largest_source_max_ratio",
        type=float,
        default=2.0,
        help="Largest source is capped to this ratio times the second-largest source.",
    )
    parser.add_argument(
        "--largest_source_hard_cap",
        type=int,
        default=None,
        help="Optional absolute cap on the largest source sample count after truncation.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    parser.add_argument("--experiment_name", default="dataset_source_cnn")
    parser.add_argument("--checkpoint_dir", default="checkpoints/dataset_source_cnn")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    parser.add_argument(
        "--disable_class_weights",
        action="store_true",
        help="Disable inverse-frequency class weighting for the source classes.",
    )
    parser.add_argument("--fast_dev_run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.largest_source_max_ratio <= 0:
        raise ValueError("--largest_source_max_ratio must be > 0.")
    if args.largest_source_hard_cap is not None and args.largest_source_hard_cap <= 0:
        raise ValueError("--largest_source_hard_cap must be > 0 when provided.")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    datamodule = DatasetSourceDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_rim_one=not args.exclude_rim_one,
        selected_sources=args.sources,
        truncate_largest_source=not args.disable_largest_truncation,
        largest_source_max_ratio=args.largest_source_max_ratio,
        largest_source_hard_cap=args.largest_source_hard_cap,
    )

    datamodule.setup(stage="fit")

    class_weights = None if args.disable_class_weights else datamodule.class_weights
    if class_weights is None:
        print("[INFO] Class weights disabled.")

    model = SourceDatasetCNN(
        num_classes=len(datamodule.class_names),
        class_names=datamodule.class_names,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        class_weights=class_weights,
    )

    logger = CSVLogger(save_dir="lightning_logs", name=args.experiment_name)
    print(f"[INFO] Logging to: {logger.log_dir}")

    datamodule.export_split_artifacts(Path(logger.log_dir))

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="source_cnn-{epoch:02d}-{val_f1_macro:.4f}",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_f1_macro",
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
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    if args.fast_dev_run:
        print("[INFO] fast_dev_run enabled: skipping final test pass.")
        return

    print("\n" + "=" * 70)
    print("Testing with best checkpoint on held-out test split")
    print("=" * 70)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
