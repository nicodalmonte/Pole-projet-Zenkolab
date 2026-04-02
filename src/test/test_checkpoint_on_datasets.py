"""Evaluate a trained checkpoint on one or more datasets.

Usage examples:
    python test/test_checkpoint_on_datasets.py \
        --checkpoint checkpoints/version_0/last.ckpt \
        --datasets REFUGE2 G1020:test LAG:validation

Dataset spec format:
    DATASET_NAME[:split]

If split is omitted, defaults are:
    - ACRIMA: train
    - FUNDUS: validation
    - LAG: test
    - ORIGA: train
    - REFUGE2: test
    - JRAIGS: train
    - G1020: test
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import (  # noqa: E402
    ACRIMADataset,
    FundusTrainValDataset,
    G1020Dataset,
    JRAIGSDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
)
from src.train.train import build_transforms  # noqa: E402


DATASET_REGISTRY = {
    "ACRIMA": ACRIMADataset,
    "FUNDUS": FundusTrainValDataset,
    "LAG": LAGDataset,
    "ORIGA": ORIGADataset,
    "REFUGE2": REFUGE2Dataset,
    "JRAIGS": JRAIGSDataset,
    "G1020_TEST": G1020Dataset,
    "G1020_TRAIN": G1020Dataset,
}

DEFAULT_SPLITS = {
    "REFUGE2": "train",
    "G1020_TEST": "test",
    "G1020_TRAIN": "train",
    "ACRIMA" : "test",
}

MODEL_REGISTRY = {
    "dinov3_1": "src.models.dino_v3_1:DinoV3_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DinoV3_1 checkpoint on one or more datasets."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model",
        default="dinov3_1",
        help=(
            "Model to load. Use a built-in alias (dinov3_1) or a full "
            "'module.path:ClassName' value for custom models."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="data/datasets",
        help="Root datasets directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["REFUGE2"],
        help=(
            "Dataset specs in the form NAME[:split]. "
            "Example: REFUGE2:test LAG:validation G1020:test"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument(
        "--devices",
        default="1",
        help="Number of devices or 'auto'.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately if one dataset evaluation fails.",
    )
    return parser.parse_args()


def resolve_model_class(model_spec: str):
    spec = MODEL_REGISTRY.get(model_spec.lower(), model_spec)
    if ":" not in spec:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Invalid --model '{model_spec}'. Use one of [{valid}] "
            "or provide 'module.path:ClassName'."
        )

    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ValueError(
            f"Model class '{class_name}' not found in module '{module_name}'."
        )
    return model_class


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    name, split = (spec.split(":", 1) + [None])[:2]
    dataset_name = name.strip().upper()

    if dataset_name not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid values: {valid}")

    if split is None or not split.strip():
        split = DEFAULT_SPLITS[dataset_name]
    return dataset_name, split.strip()


def build_dataloader(
    dataset_name: str,
    split: str,
    data_dir: str,
    eval_transform,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(data_dir=data_dir, split=split, transforms=eval_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_class = resolve_model_class(args.model)
    model = model_class.load_from_checkpoint(str(ckpt_path), map_location="cpu")

    backbone_name = str(getattr(model.hparams, "backbone_name"))
    image_size = int(getattr(model.hparams, "img_size"))
    _, eval_transform = build_transforms(backbone_name=backbone_name, image_size=image_size)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
    )

    print("\n" + "=" * 72)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model     : {model_class.__module__}:{model_class.__name__}")
    print(f"Backbone  : {backbone_name}")
    print(f"Image size: {image_size}")
    print("=" * 72)

    failures = 0

    for spec in args.datasets:
        try:
            dataset_name, split = parse_dataset_spec(spec)
            dataloader = build_dataloader(
                dataset_name=dataset_name,
                split=split,
                data_dir=args.data_dir,
                eval_transform=eval_transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            print(f"\n>>> Testing on {dataset_name} (split={split}, n={len(dataloader.dataset)})")
            results = trainer.test(model, dataloaders=dataloader, verbose=True)
            metrics = results[0] if results else {}

            if not metrics:
                print("No metrics returned.")
                continue

            for key in sorted(metrics):
                print(f"{key}: {metrics[key]:.6f}")

        except Exception as exc:
            failures += 1
            print(f"\n[ERROR] Failed on '{spec}': {exc}")
            if args.fail_fast:
                raise

    print("\n" + "=" * 72)
    if failures == 0:
        print("Evaluation finished without dataset failures.")
    else:
        print(f"Evaluation finished with {failures} dataset failure(s).")
    print("=" * 72)


if __name__ == "__main__":
    main()
