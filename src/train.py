"""
Training script for RETFound / ViT glaucoma classifier.

Usage:
    python src/train.py --epochs 30 --batch_size 32          # entraînement simple
    python src/train.py --epochs 30 --mlflow                 # + suivi MLflow
    python src/train.py --model vit --epochs 30 --mlflow

MLflow (optionnel, activé avec --mlflow) :
    Les runs sont stockés dans ./mlruns/ (local, aucun cloud).
    Pour voir le dashboard : mlflow ui   → http://127.0.0.1:5000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import GlaucomaDataset, build_transforms, build_train_dataset, build_val_dataset
from model import build_retfound, build_vit

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ViT/RETFound for glaucoma detection")
    # Model
    p.add_argument("--model",          type=str,   default="retfound",
                   choices=["vit", "retfound"],
                   help="vit = ViT-Large ImageNet-21k | retfound = RETFound MAE retinal")
    # Data
    p.add_argument("--data_root",      type=str,   default="datasets/",
                   help="Root folder containing all dataset symlinks")
    p.add_argument("--test_dir",       type=str,   default="datasets/REFUGE2/train",
                   help="External test set (REFUGE2).")
    # Training
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--patience",       type=int,   default=7)
    p.add_argument("--freeze_epochs",  type=int,   default=3)
    p.add_argument("--mlflow",         action="store_true", default=False,
                   help="Enable MLflow experiment tracking")
    p.add_argument("--mlflow_exp",     type=str,   default="glaucoma-detection")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=scaler is not None):
            logits = model(images)
            loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    n    = len(all_labels)
    avg_loss = total_loss / n
    auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc  = accuracy_score(all_labels, preds)

    return {"loss": avg_loss, "auc": auc, "acc": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- Datasets ---
    train_dataset = build_train_dataset(args.data_root)
    val_dataset   = build_val_dataset(args.data_root)
    test_dataset  = GlaucomaDataset(args.test_dir, split="test")
    print(f"[train] Test samples (REFUGE2): {test_dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- Model ---
    build_fn = build_retfound if args.model == "retfound" else build_vit
    model    = build_fn(num_classes=2, freeze_backbone=True).to(device)

    # --- Class-weighted loss ---
    class_weights = train_dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer (head only during warm-up) ---
    head_params = list(model.head.parameters())
    optimizer = AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler() if device.type == "cuda" else None

    # --- MLflow : ouvre un run et enregistre tous les hyperparamètres ---
    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.mlflow_exp)   # crée l'expérience si elle n'existe pas
        run_name = f"{args.model}_e{args.epochs}_bs{args.batch_size}_lr{args.lr:.0e}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(args))            # sauvegarde tous les args d'un coup
        print(f"[MLflow] exp='{args.mlflow_exp}'  run='{run_name}'")
        print(f"[MLflow] dashboard → mlflow ui  (http://127.0.0.1:5000)")
    elif args.mlflow and not MLFLOW_AVAILABLE:
        print("[warn] MLflow non installé. Lance: uv pip install mlflow")

    # --- Training loop ---
    best_auc       = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after freeze_epochs warm-up
        if epoch == args.freeze_epochs + 1:
            print("[train] Unfreezing backbone …")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
                    {"params": model.head.parameters(),     "lr": args.lr},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6
            )

        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_stats   = run_epoch(model, val_loader,   criterion, None,      None,   device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| Train loss={train_stats['loss']:.4f} auc={train_stats['auc']:.4f} acc={train_stats['acc']:.4f} "
            f"| Val   loss={val_stats['loss']:.4f}  auc={val_stats['auc']:.4f}  acc={val_stats['acc']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if args.mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics({f"train/{k}": v for k, v in train_stats.items()}, step=epoch)
            mlflow.log_metrics({f"val/{k}": v for k, v in val_stats.items()}, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)

        # --- Checkpoint ---
        if val_stats["auc"] > best_auc:
            best_auc = val_stats["auc"]
            patience_count = 0
            ckpt_path = CHECKPOINT_DIR / f"best_{args.model}.pth"
            torch.save(
                {
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "best_auc":   best_auc,
                    "args":       vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model (val AUC={best_auc:.4f}) → {ckpt_path}")
            
            # Log checkpoint to MLflow
            if args.mlflow and MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[train] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\n[train] Done! Best validation AUC: {best_auc:.4f}")

    # --- Final evaluation on REFUGE2 test set ---
    from evaluate import evaluate as eval_fn
    from torch.utils.data import DataLoader as DL
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    print("\n[train] Running final evaluation on REFUGE2 test set …")
    ckpt = torch.load(CHECKPOINT_DIR / f"best_{args.model}.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    def _collate(batch):
        imgs   = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        return imgs, labels, paths

    test_loader = DL(test_dataset, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=True, collate_fn=_collate)
    labels_arr, probs_arr, _ = eval_fn(model, test_loader, device)
    preds = (probs_arr >= 0.5).astype(int)
    auc  = roc_auc_score(labels_arr, probs_arr)
    acc  = accuracy_score(labels_arr, preds)
    tn, fp, fn, tp = confusion_matrix(labels_arr, preds).ravel()
    print(f"[REFUGE2 test] AUC={auc:.4f} | Acc={acc:.4f} | "
          f"Sensitivity={tp/(tp+fn):.4f} | Specificity={tn/(tn+fp):.4f}")

    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metrics({"test/auc": auc, "test/acc": acc})
        mlflow.log_metric("test/sensitivity", tp/(tp+fn))
        mlflow.log_metric("test/specificity", tn/(tn+fp))
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.end_run()
        print(f"\n✓ MLflow run saved! View with: mlflow ui")


if __name__ == "__main__":
    main()
