"""DINOv3-Small generalisation experiment — v3.

Changes vs v2:
  - Condition A = mixed pool (JRAIGS + ACRIMA + ORIGA + LAG + Harvard),
    glaucoma-oversampled: 1200 pos + 800 neg = 2000 total
  - Condition B = JRAIGS-only, 1000 pos + 1000 neg = 2000 (was condition A in v2)
  - Condition C = ACRIMA + ORIGA + LAG (unchanged from v2)
  - Condition D = Harvard-only (unchanged from v2)
  - Test sets: all RIMone (train+test) + all AIRROGS + all Fundus (train+val)
    => AIRROGS and Fundus are no longer in the training pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler,
)
from torchvision import transforms
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.datasets import (
    ACRIMADataset,
    ORIGADataset,
    LAGDataset,
    AIROGSLightDataset,
    FundusTrainValDataset,
    HarvardGlaucomaDataset,
    JRAIGSDataset,
    REFUGE2Dataset,
)
from src.datasets.RIMONE import RIMONEDataset
from src.models.dino_v3_1 import DinoV3_1

SEED = 42
BACKBONE = "vit_small_patch16_dinov3.lvd1689m"

# Condition B (JRAIGS-only): balanced 50/50
JRAIGS_N_PER_CLASS = 1_000   # 1000 pos + 1000 neg = 2000

# Condition A (mixed): glaucoma-oversampled
MIX_A_N_POS = 1_200          # 1200 glaucoma
MIX_A_N_NEG = 800            # 800 normal  → 2000 total, 60% glaucoma

# Conditions C and D: balanced 50/50
POOL_N_PER_CLASS = 1_000     # 1000 pos + 1000 neg = 2000


# ---------------------------------------------------------------------------
# Model — same as v2
# ---------------------------------------------------------------------------

class DinoV3_1_V2(DinoV3_1):
    """DinoV3_1 with label smoothing, differential LR and CosineAnnealing."""

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = F.cross_entropy(
            logits, batch["label"],
            weight=self.loss_weight,
            label_smoothing=0.1,
        )
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.train_auc.update(probs, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.head.parameters())

        if self._backbone_is_frozen:
            param_groups = [{"params": head_params, "lr": self.hparams.lr}]
        else:
            param_groups = [
                {"params": backbone_params, "lr": self.hparams.lr * 0.05},
                {"params": head_params,     "lr": self.hparams.lr},
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=25,
            eta_min=self.hparams.lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Harvard binary wrapper
# ---------------------------------------------------------------------------

class BinaryMappedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        sample["label"] = torch.tensor(min(int(sample["label"]), 1), dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Fast label extraction
# ---------------------------------------------------------------------------

def get_labels_fast(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent = get_labels_fast(dataset.dataset)
        return [parent[i] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        out: list[int] = []
        for ds in dataset.datasets:
            out.extend(get_labels_fast(ds))
        return out

    if isinstance(dataset, BinaryMappedDataset):
        return [min(int(l), 1) for l in get_labels_fast(dataset.dataset)]

    if hasattr(dataset, "labels"):
        return [int(l) for l in dataset.labels]  # type: ignore[attr-defined]

    if hasattr(dataset, "samples"):
        return [int(s[1]) for s in dataset.samples]  # type: ignore[attr-defined]

    if isinstance(dataset, ACRIMADataset):
        return [1 if "_g_" in Path(p).name else 0 for p in dataset.image_paths]

    if isinstance(dataset, LAGDataset):
        return [1 if Path(p).name.startswith("g.") else 0 for p in dataset.image_paths]

    raise ValueError(f"Cannot extract labels from {type(dataset).__name__}")


def class_counts(labels: list[int]) -> tuple[int, int]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return n_neg, n_pos


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int = 224):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, img_size, img_size)
    eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

    augment = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    train_tf = transforms.Compose([
        augment,
        eval_tf,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Subsampling helpers
# ---------------------------------------------------------------------------

def _subsample(dataset: Dataset, n_pos: int, n_neg: int, seed: int) -> Subset:
    """Return a Subset with at most n_pos positives and n_neg negatives."""
    labels = get_labels_fast(dataset)
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    g = torch.Generator().manual_seed(seed)
    actual_pos = min(n_pos, len(pos_idx))
    actual_neg = min(n_neg, len(neg_idx))

    sel_pos = torch.randperm(len(pos_idx), generator=g)[:actual_pos].tolist()
    sel_neg = torch.randperm(len(neg_idx), generator=g)[:actual_neg].tolist()
    selected = [pos_idx[i] for i in sel_pos] + [neg_idx[i] for i in sel_neg]

    print(f"    Subsample: neg={actual_neg}  pos={actual_pos}  total={len(selected)}")
    return Subset(dataset, selected)


def _stratified_subsample(dataset: Dataset, n_per_class: int, seed: int) -> Subset:
    return _subsample(dataset, n_per_class, n_per_class, seed)


def _train_val_split(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = int(n * (1.0 - val_ratio))
    return perm[:k], perm[k:]


def _make_dl(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _make_balanced_dl(
    ds: Dataset,
    labels: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    n_neg, n_pos = class_counts(labels)
    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: degenerate class distribution (neg={n_neg}, pos={n_pos}), using shuffle")
        return _make_dl(ds, batch_size, num_workers, shuffle=True)

    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = torch.tensor([w_pos if l == 1 else w_neg for l in labels])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    print(f"  Class balance  neg={n_neg}  pos={n_pos}  (WeightedRandomSampler)")
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _split_and_build_loaders(
    train_ds: Subset,
    eval_ds: Subset,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    label: str,
) -> tuple[DataLoader, DataLoader]:
    n = len(train_ds)  # type: ignore[arg-type]
    train_idx, val_idx = _train_val_split(n, val_ratio, SEED)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(eval_ds,  val_idx)

    train_labels = get_labels_fast(train_subset)
    print(f"  {label} → train={len(train_idx)}  val={len(val_idx)}")

    train_dl = _make_balanced_dl(train_subset, train_labels, batch_size, num_workers)
    val_dl   = _make_dl(val_subset, batch_size, num_workers, shuffle=False)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------

def build_mixed_glaucoma_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    """Condition A: mixed pool (JRAIGS+ACRIMA+ORIGA+LAG+Harvard),
    1200 pos + 800 neg = 2000. AIRROGS and Fundus excluded (they are test sets).
    """
    print(f"\n--- Condition A: Mixed glaucoma-oversampled (v3 — {MIX_A_N_POS}pos+{MIX_A_N_NEG}neg) ---")
    print("  NOTE: AIRROGS and Fundus excluded from pool (reserved as test sets)")

    def try_load_pair(name, fn_train, fn_eval, acc_train, acc_eval):
        try:
            ds_train = fn_train()
            ds_eval  = fn_eval()
            labels = get_labels_fast(ds_train)
            n_neg, n_pos = class_counts(labels)
            print(f"  Pool | {name}: {len(ds_train)} samples  (neg={n_neg} pos={n_pos})")  # type: ignore[arg-type]
            acc_train.append(ds_train)
            acc_eval.append(ds_eval)
        except Exception as e:
            print(f"  Pool | {name}: SKIPPED ({e})")

    candidates_train: list[Dataset] = []
    candidates_eval:  list[Dataset] = []

    try_load_pair(
        "JRAIGS",
        lambda: JRAIGSDataset(data_dir=data_dir, transforms=train_tf),
        lambda: JRAIGSDataset(data_dir=data_dir, transforms=eval_tf),
        candidates_train, candidates_eval,
    )
    try_load_pair(
        "ACRIMA",
        lambda: ACRIMADataset(data_dir=data_dir, transforms=train_tf),
        lambda: ACRIMADataset(data_dir=data_dir, transforms=eval_tf),
        candidates_train, candidates_eval,
    )
    try_load_pair(
        "ORIGA",
        lambda: ORIGADataset(data_dir=data_dir, transforms=train_tf),
        lambda: ORIGADataset(data_dir=data_dir, transforms=eval_tf),
        candidates_train, candidates_eval,
    )
    try_load_pair(
        "LAG(train)",
        lambda: LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
        lambda: LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf),
        candidates_train, candidates_eval,
    )
    try_load_pair(
        "Harvard(binary)",
        lambda: BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=train_tf)),
        lambda: BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=eval_tf)),
        candidates_train, candidates_eval,
    )

    if not candidates_train:
        raise RuntimeError("No datasets could be loaded for condition A.")

    combined_train = ConcatDataset(candidates_train)
    combined_eval  = ConcatDataset(candidates_eval)

    labels_all = get_labels_fast(combined_train)
    n_neg_all, n_pos_all = class_counts(labels_all)
    print(f"  Combined pool: {len(combined_train)} total  (neg={n_neg_all} pos={n_pos_all})")  # type: ignore[arg-type]

    sub_train = _subsample(combined_train, MIX_A_N_POS, MIX_A_N_NEG, SEED)
    sub_eval  = _subsample(combined_eval,  MIX_A_N_POS, MIX_A_N_NEG, SEED)

    return _split_and_build_loaders(
        sub_train, sub_eval, val_ratio, batch_size, num_workers, "Mixed-glaucoma"
    )


def build_jraigs_only_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    """Condition B: JRAIGS-only, 1000 pos + 1000 neg = 2000."""
    print("\n--- Condition B: JRAIGS-only (v3 — stratified 2k) ---")
    ds_train = JRAIGSDataset(data_dir=data_dir, transforms=train_tf)
    ds_eval  = JRAIGSDataset(data_dir=data_dir, transforms=eval_tf)
    labels_full = get_labels_fast(ds_train)
    n_neg_full, n_pos_full = class_counts(labels_full)
    print(f"  JRAIGS: {len(ds_train)} total  (neg={n_neg_full} pos={n_pos_full})")  # type: ignore[arg-type]
    sub_train = _stratified_subsample(ds_train, JRAIGS_N_PER_CLASS, SEED)
    sub_eval  = _stratified_subsample(ds_eval,  JRAIGS_N_PER_CLASS, SEED)
    return _split_and_build_loaders(
        sub_train, sub_eval, val_ratio, batch_size, num_workers, "JRAIGS-only"
    )


def build_acrima_origa_lag_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    """Condition C: ACRIMA + ORIGA + LAG, 1000 pos + 1000 neg = 2000."""
    print("\n--- Condition C: ACRIMA + ORIGA + LAG (v3 — stratified 2k) ---")
    candidates_train: list[Dataset] = []
    candidates_eval:  list[Dataset] = []

    def try_load_pair(name: str, fn_train, fn_eval):
        try:
            ds_train = fn_train()
            ds_eval  = fn_eval()
            labels = get_labels_fast(ds_train)
            n_neg, n_pos = class_counts(labels)
            print(f"  {name}: {len(ds_train)} total  (neg={n_neg} pos={n_pos})")  # type: ignore[arg-type]
            candidates_train.append(ds_train)
            candidates_eval.append(ds_eval)
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    try_load_pair(
        "ACRIMA",
        lambda: ACRIMADataset(data_dir=data_dir, transforms=train_tf),
        lambda: ACRIMADataset(data_dir=data_dir, transforms=eval_tf),
    )
    try_load_pair(
        "ORIGA",
        lambda: ORIGADataset(data_dir=data_dir, transforms=train_tf),
        lambda: ORIGADataset(data_dir=data_dir, transforms=eval_tf),
    )
    try_load_pair(
        "LAG(train)",
        lambda: LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
        lambda: LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf),
    )

    if not candidates_train:
        raise RuntimeError("No ACRIMA/ORIGA/LAG datasets could be loaded.")

    combined_train = ConcatDataset(candidates_train)
    combined_eval  = ConcatDataset(candidates_eval)
    sub_train = _stratified_subsample(combined_train, POOL_N_PER_CLASS, SEED)
    sub_eval  = _stratified_subsample(combined_eval,  POOL_N_PER_CLASS, SEED)
    return _split_and_build_loaders(
        sub_train, sub_eval, val_ratio, batch_size, num_workers, "ACRIMA+ORIGA+LAG"
    )


def build_harvard_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    """Condition D: Harvard-only, up to 1000 pos + 1000 neg = 2000."""
    print("\n--- Condition D: Harvard-only (v3 — stratified 2k) ---")
    try:
        ds_train = BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=train_tf))
        ds_eval  = BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=eval_tf))
    except Exception as e:
        raise RuntimeError(f"Harvard dataset could not be loaded: {e}") from e

    labels = get_labels_fast(ds_train)
    n_neg, n_pos = class_counts(labels)
    print(f"  Harvard: {len(ds_train)} total  (neg={n_neg} pos={n_pos})")  # type: ignore[arg-type]

    sub_train = _stratified_subsample(ds_train, POOL_N_PER_CLASS, SEED)
    sub_eval  = _stratified_subsample(ds_eval,  POOL_N_PER_CLASS, SEED)
    return _split_and_build_loaders(
        sub_train, sub_eval, val_ratio, batch_size, num_workers, "Harvard-only"
    )


def build_test_dataloaders(data_dir: str, eval_tf, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    """Test sets: all RIMone (train+test) + all AIRROGS + all Fundus (train+val).
    None of these appear in training conditions A, B, C, D.
    """
    print("\n--- Test datasets (held-out, zero-shot) ---")
    factories = {
        "RIMONE(train)":    lambda: RIMONEDataset(data_dir=data_dir, split="train",       transforms=eval_tf),
        "RIMONE(test)":     lambda: RIMONEDataset(data_dir=data_dir, split="test",        transforms=eval_tf),
        "AIRROGS":          lambda: AIROGSLightDataset(data_dir=data_dir,                 transforms=eval_tf),
        "Fundus(train)":    lambda: FundusTrainValDataset(data_dir=data_dir, split="train",      transforms=eval_tf),
        "Fundus(val)":      lambda: FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
    }
    dls: dict[str, DataLoader] = {}
    for name, fn in factories.items():
        try:
            ds = fn()
            labels = get_labels_fast(ds)
            n_neg, n_pos = class_counts(labels)
            dls[name] = _make_dl(ds, batch_size, num_workers)
            print(f"  {name}: {len(ds)} samples  (neg={n_neg} pos={n_pos})")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")
    return dls


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_phase(
    model: DinoV3_1_V2,
    train_dl: DataLoader,
    val_dl: DataLoader,
    max_epochs: int,
    log_name: str,
    ckpt_dir: str,
    precision: str,
    early_stop_patience: int = 8,
) -> tuple[str, str]:
    logger = CSVLogger(save_dir="lightning_logs/generalization_v3", name=log_name)
    v = logger.version

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{log_name}_v{v}" + "-{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc", mode="max",
            save_top_k=1, save_last=False,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=early_stop_patience, min_delta=1e-3, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
    ]
    trainer = L.Trainer(
        max_epochs=max_epochs, callbacks=callbacks,
        precision=precision, logger=logger,
        log_every_n_steps=10, deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl)
    best = trainer.checkpoint_callback.best_model_path
    print(f"  Best checkpoint: {best}")
    return best, logger.log_dir


def train_condition(
    label: str,
    train_dl: DataLoader,
    val_dl: DataLoader,
    args,
    ckpt_dir: str,
    figures_dir: str,
) -> tuple[str, DataLoader]:
    slug = label.lower().replace(" ", "_").replace("-", "_").replace("+", "_")

    print("\n" + "=" * 60)
    print(f"[{label}] Phase 1 — backbone frozen, lr={args.lr1}")
    print("=" * 60)
    model_p1 = DinoV3_1_V2(
        backbone_name=BACKBONE, pretrained=True,
        lr=args.lr1, img_size=args.img_size,
        dropout=0.35,
        weight_decay=5e-3,
        unfreeze_backbone_epoch=args.max_epochs + 1,
    )
    best1, log_dir1 = run_phase(
        model_p1, train_dl, val_dl,
        max_epochs=args.max_epochs,
        log_name=f"{slug}_phase1",
        ckpt_dir=ckpt_dir,
        precision=args.precision,
        early_stop_patience=8,
    )
    plot_training_curves(log_dir1, f"{label} — Phase 1", figures_dir)

    print("\n" + "=" * 60)
    print(f"[{label}] Phase 2 — backbone unfrozen, backbone_lr={args.lr2 * 0.05:.2e}, head_lr={args.lr2}")
    print("=" * 60)
    model_p2 = DinoV3_1_V2.load_from_checkpoint(
        best1,
        lr=args.lr2,
        weight_decay=5e-3,
        unfreeze_backbone_epoch=0,
    )
    best2, log_dir2 = run_phase(
        model_p2, train_dl, val_dl,
        max_epochs=args.max_epochs,
        log_name=f"{slug}_phase2",
        ckpt_dir=ckpt_dir,
        precision=args.precision,
        early_stop_patience=5,
    )
    plot_training_curves(log_dir2, f"{label} — Phase 2", figures_dir)
    return best2, val_dl


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_probs_labels(model: torch.nn.Module, dl: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    all_probs, all_labels = [], []
    model.eval()
    for batch in dl:
        logits = model(batch["image"].to(device))
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.cpu())
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def _find_optimal_threshold(probs: torch.Tensor, labels: torch.Tensor) -> float:
    thresholds = torch.linspace(0.01, 0.99, 300)
    best_j, best_t = -1.0, 0.5
    for t in thresholds:
        preds = (probs >= t).long()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        j = (sens + spec - 1).item()
        if j > best_j:
            best_j, best_t = j, t.item()
    return best_t


def _metrics_at_threshold(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, float]:
    from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity
    auc  = BinaryAUROC()(probs, labels).item()
    acc  = BinaryAccuracy(threshold=threshold)(probs, labels).item()
    f1   = BinaryF1Score(threshold=threshold)(probs, labels).item()
    sens = BinaryRecall(threshold=threshold)(probs, labels).item()
    spec = BinarySpecificity(threshold=threshold)(probs, labels).item()
    return {"test_auc": auc, "test_acc": acc, "test_f1": f1,
            "test_sensitivity": sens, "test_specificity": spec}


def evaluate_all(
    ckpt_path: str,
    val_dl: DataLoader,
    test_dls: dict[str, DataLoader],
    precision: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DinoV3_1_V2.load_from_checkpoint(ckpt_path).to(device)

    val_probs, val_labels = _collect_probs_labels(m, val_dl)
    threshold = _find_optimal_threshold(val_probs, val_labels)
    print(f"  Optimal threshold (Youden's J on val): {threshold:.3f}")

    results: dict[str, dict] = {}
    for ds_name, dl in test_dls.items():
        probs, labels = _collect_probs_labels(m, dl)
        r = _metrics_at_threshold(probs, labels, threshold)
        results[ds_name] = r
        print(
            f"  {ds_name:<22}  AUC={r['test_auc']:.4f}  "
            f"Acc={r['test_acc']:.4f}  F1={r['test_f1']:.4f}  "
            f"Sens={r['test_sensitivity']:.4f}  Spec={r['test_specificity']:.4f}  "
            f"(thr={threshold:.3f})"
        )
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_training_curves(log_dir: str, phase_label: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = Path(log_dir) / "metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df_ep = df.dropna(subset=["val_auc"]).copy()
    if df_ep.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss_epoch" in df_ep.columns:
        ax1.plot(df_ep["epoch"], df_ep["train_loss_epoch"], label="Train loss")
    if "val_loss" in df_ep.columns:
        ax1.plot(df_ep["epoch"], df_ep["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{phase_label} — Loss"); ax1.legend()

    if "train_auc" in df_ep.columns:
        ax2.plot(df_ep["epoch"], df_ep["train_auc"], linestyle="--", label="Train AUC", alpha=0.7)
    ax2.plot(df_ep["epoch"], df_ep["val_auc"], color="darkorange", label="Val AUC")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC-ROC")
    ax2.set_title(f"{phase_label} — AUC"); ax2.legend(); ax2.set_ylim(0, 1)

    plt.tight_layout()
    slug = "".join(c if c.isalnum() else "_" for c in phase_label.lower())
    out = Path(out_dir) / f"{slug}_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_full_metrics(results: dict, condition_label: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ds_names = list(results.keys())
    metrics = ["test_auc", "test_acc", "test_f1", "test_sensitivity", "test_specificity"]
    labels  = ["AUC",      "Accuracy", "F1",      "Sensitivity",      "Specificity"]
    colors  = ["#2196F3",  "#4CAF50",  "#FF9800", "#E91E63",          "#9C27B0"]

    x = np.arange(len(ds_names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(ds_names)), 6))
    for i, (metric, lbl, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[ds].get(metric, 0.0) for ds in ds_names]
        ax.bar(x + i * w, vals, w, label=lbl, color=color)

    ax.set_xlabel("Test dataset"); ax.set_ylabel("Score")
    ax.set_title(f"DINOv3-Small v3 [{condition_label}] — Zero-shot test metrics")
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(ds_names, rotation=30, ha="right")
    ax.legend(loc="upper right"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    slug = "".join(c if c.isalnum() else "_" for c in condition_label.lower())
    out = Path(out_dir) / f"{slug}_test_metrics_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DINOv3-Small generalisation experiment v3")
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_epochs",  type=int,   default=25)
    p.add_argument("--lr1",         type=float, default=1e-3)
    p.add_argument("--lr2",         type=float, default=1e-4)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints/generalization_v3")
    p.add_argument("--figures_dir", default="figures/generalization_v3")
    p.add_argument("--precision",   default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--condition",   default="all",
                   choices=["a", "b", "c", "d", "ab", "cd", "all"],
                   help=(
                       "a=Mixed-glaucoma, b=JRAIGS-only, "
                       "c=ACRIMA+ORIGA+LAG, d=Harvard, "
                       "ab=a+b, cd=c+d, all=a+b+c+d"
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    L.seed_everything(SEED, workers=True)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)

    test_dls = build_test_dataloaders(
        args.data_dir, eval_tf, args.batch_size, args.num_workers,
    )

    results: dict[str, dict] = {}

    # ----------------------------------------------------------------
    # Condition A — Mixed glaucoma-oversampled
    # ----------------------------------------------------------------
    if args.condition in ("a", "ab", "all"):
        train_dl_a, val_dl_a = build_mixed_glaucoma_dataloaders(
            args.data_dir, train_tf, eval_tf,
            args.batch_size, args.num_workers, args.val_ratio,
        )
        ckpt_a, val_dl_a_cal = train_condition(
            label="Mixed-glaucoma",
            train_dl=train_dl_a, val_dl=val_dl_a, args=args,
            ckpt_dir=str(Path(args.ckpt_dir) / "mixed_glaucoma"),
            figures_dir=args.figures_dir,
        )
        print("\n" + "=" * 60)
        print("Condition A — Mixed-glaucoma — Zero-shot evaluation")
        print("=" * 60)
        results["a"] = evaluate_all(ckpt_a, val_dl_a_cal, test_dls, args.precision)
        plot_full_metrics(results["a"], "Mixed-glaucoma", args.figures_dir)
        with open(Path(args.figures_dir) / "test_results_v3_a.json", "w") as f:
            json.dump({"mixed_glaucoma": results["a"]}, f, indent=2)

    # ----------------------------------------------------------------
    # Condition B — JRAIGS-only
    # ----------------------------------------------------------------
    if args.condition in ("b", "ab", "all"):
        train_dl_b, val_dl_b = build_jraigs_only_dataloaders(
            args.data_dir, train_tf, eval_tf,
            args.batch_size, args.num_workers, args.val_ratio,
        )
        ckpt_b, val_dl_b_cal = train_condition(
            label="JRAIGS-only",
            train_dl=train_dl_b, val_dl=val_dl_b, args=args,
            ckpt_dir=str(Path(args.ckpt_dir) / "jraigs_only"),
            figures_dir=args.figures_dir,
        )
        print("\n" + "=" * 60)
        print("Condition B — JRAIGS-only — Zero-shot evaluation")
        print("=" * 60)
        results["b"] = evaluate_all(ckpt_b, val_dl_b_cal, test_dls, args.precision)
        plot_full_metrics(results["b"], "JRAIGS-only", args.figures_dir)
        with open(Path(args.figures_dir) / "test_results_v3_b.json", "w") as f:
            json.dump({"jraigs_only": results["b"]}, f, indent=2)

    # ----------------------------------------------------------------
    # Condition C — ACRIMA + ORIGA + LAG
    # ----------------------------------------------------------------
    if args.condition in ("c", "cd", "all"):
        train_dl_c, val_dl_c = build_acrima_origa_lag_dataloaders(
            args.data_dir, train_tf, eval_tf,
            args.batch_size, args.num_workers, args.val_ratio,
        )
        ckpt_c, val_dl_c_cal = train_condition(
            label="ACRIMA+ORIGA+LAG",
            train_dl=train_dl_c, val_dl=val_dl_c, args=args,
            ckpt_dir=str(Path(args.ckpt_dir) / "acrima_origa_lag"),
            figures_dir=args.figures_dir,
        )
        print("\n" + "=" * 60)
        print("Condition C — ACRIMA+ORIGA+LAG — Zero-shot evaluation")
        print("=" * 60)
        results["c"] = evaluate_all(ckpt_c, val_dl_c_cal, test_dls, args.precision)
        plot_full_metrics(results["c"], "ACRIMA+ORIGA+LAG", args.figures_dir)
        with open(Path(args.figures_dir) / "test_results_v3_c.json", "w") as f:
            json.dump({"acrima_origa_lag": results["c"]}, f, indent=2)

    # ----------------------------------------------------------------
    # Condition D — Harvard-only
    # ----------------------------------------------------------------
    if args.condition in ("d", "cd", "all"):
        train_dl_d, val_dl_d = build_harvard_dataloaders(
            args.data_dir, train_tf, eval_tf,
            args.batch_size, args.num_workers, args.val_ratio,
        )
        ckpt_d, val_dl_d_cal = train_condition(
            label="Harvard-only",
            train_dl=train_dl_d, val_dl=val_dl_d, args=args,
            ckpt_dir=str(Path(args.ckpt_dir) / "harvard_only"),
            figures_dir=args.figures_dir,
        )
        print("\n" + "=" * 60)
        print("Condition D — Harvard-only — Zero-shot evaluation")
        print("=" * 60)
        results["d"] = evaluate_all(ckpt_d, val_dl_d_cal, test_dls, args.precision)
        plot_full_metrics(results["d"], "Harvard-only", args.figures_dir)
        with open(Path(args.figures_dir) / "test_results_v3_d.json", "w") as f:
            json.dump({"harvard_only": results["d"]}, f, indent=2)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY — AUC across all conditions [v3]")
        print("=" * 60)
        cond_labels = {"a": "Mixed-glaucoma", "b": "JRAIGS-only", "c": "ACRIMA+ORIGA+LAG", "d": "Harvard-only"}
        all_ds = sorted(set(ds for r in results.values() for ds in r))
        header = f"  {'Dataset':<22}  " + "  ".join(f"{cond_labels[k]:<18}" for k in sorted(results))
        print(header)
        for ds in all_ds:
            row = f"  {ds:<22}  " + "  ".join(
                f"{results[k].get(ds, {}).get('test_auc', float('nan')):.4f}{'':14}"
                for k in sorted(results)
            )
            print(row)
        with open(Path(args.figures_dir) / "test_results_v3_all.json", "w") as f:
            json.dump({cond_labels[k]: v for k, v in results.items()}, f, indent=2)
        print(f"\nAll results saved to {args.figures_dir}/test_results_v3_all.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
