"""Train a ResNet classifier on ORIGA using CSV split A/B (train/val)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ORIGADataset(Dataset):
    """ORIGA dataset backed by labels_train_val.csv and an Images directory."""

    def __init__(self, csv_path: str, images_dir: str, split: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)

        print(f"Split {split}: {len(self.data)} images")
        print(f"  - Label 0 (healthy): {(self.data['label'] == 0).sum()}")
        print(f"  - Label 1 (glaucoma): {(self.data['label'] == 1).sum()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_image_path = Path(row["image_path"])
        candidate_paths = [
            raw_image_path,
            self.images_dir / raw_image_path,
            self.images_dir / raw_image_path.name,
            self.images_dir.parent / raw_image_path,
            self.images_dir.parent / raw_image_path.name,
        ]
        image_path = next((path for path in candidate_paths if path.exists()), candidate_paths[2])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def get_transforms(image_size: int = 224, is_training: bool = True):
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(model_name: str = "resnet50", num_classes: int = 2, pretrained: bool = True):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def calculate_metrics(all_labels, all_preds, all_probs):
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
    }


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    return epoch_loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORIGA ResNet with split A/B")

    default_data_dir = os.getenv(
        "ORIGA_DATA_DIR",
        "datasets/sshikamaru__glaucoma-detection/ORIGA/ORIGA",
    )
    default_csv = os.getenv("ORIGA_CSV_PATH", str(Path(default_data_dir) / "labels_train_val.csv"))
    default_images = os.getenv("ORIGA_IMAGES_DIR", str(Path(default_data_dir) / "Images"))

    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--images_dir", type=str, default=default_images)
    parser.add_argument("--train_split", type=str, default=os.getenv("ORIGA_TRAIN_SPLIT", "A"))
    parser.add_argument("--val_split", type=str, default=os.getenv("ORIGA_VAL_SPLIT", "B"))
    parser.add_argument("--output_dir", type=str, default="outputs/origa_resnet")

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ORIGADataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split=args.train_split,
        transform=get_transforms(args.image_size, is_training=True),
    )
    val_dataset = ORIGADataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split=args.val_split,
        transform=get_transforms(args.image_size, is_training=False),
    )

    train_labels = np.array(train_dataset.data["label"].tolist(), dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)
    class_counts = np.clip(class_counts, 1, None)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(args.model_name, num_classes=2, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_loss = float("inf")

    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_metrics["auc_roc"])
        history["val_auc"].append(val_metrics["auc_roc"])

        print(
            f"Train loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} auc={train_metrics['auc_roc']:.4f}"
        )
        print(
            f"Val   loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} auc={val_metrics['auc_roc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            ckpt_path = output_dir / f"best_model_{args.model_name}.pth"
            torch.save(checkpoint, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")

    with open(output_dir / f"training_history_{args.model_name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining completed.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()