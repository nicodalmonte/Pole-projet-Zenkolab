"""Evaluate one model on JRAIGS train split and save metrics to JSON.

Usage
-----
    uv run src/eval/eval_jraigs.py --model dino_large
    uv run src/eval/eval_jraigs.py --model eva
    uv run src/eval/eval_jraigs.py --model dino_small
    uv run src/eval/eval_jraigs.py --model distil_single
    uv run src/eval/eval_jraigs.py --model distil_dual
    uv run src/eval/eval_jraigs.py --model ensemble

Results are appended to results_jraigs_train.json.
Run with --summarize to print the full table from the JSON.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from src.datasets import JRAIGSDataset
from src.models.dino_v3_1 import DinoV3_1
from src.models.eva_vit import EvaViT
from src.models.student import StudentGlaucomaDistilled
from src.models.student_dual import StudentGlaucomaDualDistilled

RESULTS_DIR = Path("results_jraigs")

CKPT_DIRS = {
    "dino_large":    "checkpoints/dino_large",
    "eva":           "checkpoints/eva",
    "dino_small":    "checkpoints/dino_small",
    "distil_single": "checkpoints/student",
    "distil_dual":   "checkpoints/student_dual",
}

MODEL_NAMES = {
    "dino_large":    "DINO large standalone",
    "eva":           "EVA standalone",
    "dino_small":    "DINO small standalone",
    "distil_single": "Distil single (L→S)",
    "distil_dual":   "Distil dual (L+E→S)",
    "ensemble":      "Ensemble (L+E)",
}


def best_ckpt(directory: str) -> str:
    ckpts = [p for p in Path(directory).glob("*.ckpt") if "last" not in p.name]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return str(sorted(ckpts, key=lambda p: float(p.stem.split("val_auc=")[-1]), reverse=True)[0])


def build_transform(backbone_name: str, image_size: int):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)
    return timm.data.create_transform(**data_cfg, is_training=False)


def build_loader(transform, batch_size: int, num_workers: int, data_dir: str) -> DataLoader:
    dataset = JRAIGSDataset(data_dir=data_dir, split="train", transforms=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def init_metrics(device):
    return {
        "auc":         BinaryAUROC().to(device),
        "acc":         BinaryAccuracy().to(device),
        "f1":          BinaryF1Score().to(device),
        "sensitivity": BinaryRecall().to(device),
        "specificity": BinarySpecificity().to(device),
    }


@torch.no_grad()
def run_eval(model, loader, device) -> dict[str, float]:
    model.eval()
    m = init_metrics(device)
    for i, batch in enumerate(loader):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        m["auc"].update(probs, labels)
        m["acc"].update(preds, labels)
        m["f1"].update(preds, labels)
        m["sensitivity"].update(preds, labels)
        m["specificity"].update(preds, labels)
        if (i + 1) % 100 == 0:
            print(f"  {(i+1) * loader.batch_size} / {len(loader.dataset)} images processed", flush=True)
    return {k: v.compute().item() for k, v in m.items()}


@torch.no_grad()
def run_ensemble(dino, eva, dino_loader, eva_loader, device) -> dict[str, float]:
    dino.eval()
    eva.eval()
    m = init_metrics(device)
    for i, (db, eb) in enumerate(zip(dino_loader, eva_loader)):
        dino_imgs  = db["image"].to(device)
        eva_imgs   = eb["image"].to(device)
        labels     = db["label"].to(device)
        dino_probs = torch.softmax(dino(dino_imgs), dim=1)
        eva_probs  = torch.softmax(eva(eva_imgs), dim=1)
        avg_probs  = (dino_probs + eva_probs) / 2
        m["auc"].update(avg_probs[:, 1], labels)
        m["acc"].update(avg_probs.argmax(dim=1), labels)
        m["f1"].update(avg_probs.argmax(dim=1), labels)
        m["sensitivity"].update(avg_probs.argmax(dim=1), labels)
        m["specificity"].update(avg_probs.argmax(dim=1), labels)
        if (i + 1) % 100 == 0:
            print(f"  {(i+1) * dino_loader.batch_size} / {len(dino_loader.dataset)} images processed", flush=True)
    return {k: v.compute().item() for k, v in m.items()}


METRICS = ["auc", "acc", "f1", "sensitivity", "specificity"]


def save_result(model_key: str, metrics: dict[str, float]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{model_key}.md"
    vals = " | ".join(f"{metrics[m]:.4f}" for m in METRICS)
    out.write_text(
        f"# {MODEL_NAMES[model_key]} — JRAIGS train\n\n"
        f"| AUC | ACC | F1 | Sensitivity | Specificity |\n"
        f"|---|---|---|---|---|\n"
        f"| {vals} |\n"
    )
    print(f"Results saved to {out}")


def _parse_model_vals(key: str) -> list[str] | None:
    md = RESULTS_DIR / f"{key}.md"
    if not md.exists():
        return None
    rows = [l for l in md.read_text().splitlines() if l.startswith("|") and "---" not in l and "AUC" not in l]
    if not rows:
        return None
    return [v.strip() for v in rows[0].strip("|").split("|")]


def print_summary() -> None:
    col_w, name_w = 13, 30
    header = f"{'Modello':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in METRICS)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}\n  JRAIGS train — risultati\n{'='*len(header)}")
    print(header)
    print(sep)
    for key in MODEL_NAMES:
        name = MODEL_NAMES[key]
        vals = _parse_model_vals(key)
        if vals is None:
            print(f"{name:<{name_w}}" + "".join(f"{'—':>{col_w}}" for _ in METRICS))
        else:
            print(f"{name:<{name_w}}" + "".join(f"{float(v):>{col_w}.4f}" for v in vals))
    print(f"{'='*len(header)}\n")


def write_report(filepath: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Risultati JRAIGS train\n\n",
        f"Generato: {ts}\n\n",
        f"| Modello | AUC | ACC | F1 | Sensitivity | Specificity |\n",
        f"|---|---|---|---|---|---|\n",
    ]
    for key in MODEL_NAMES:
        name = MODEL_NAMES[key]
        vals = _parse_model_vals(key)
        if vals is None:
            lines.append(f"| {name} | — | — | — | — | — |\n")
        else:
            row = " | ".join(f"{float(v):.4f}" for v in vals)
            lines.append(f"| {name} | {row} |\n")
    Path(filepath).write_text("".join(lines))
    print(f"Report salvato in {filepath}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(CKPT_DIRS) + ["ensemble"],
                   help="Model to evaluate")
    p.add_argument("--summarize", action="store_true",
                   help="Print table from existing results and exit")
    p.add_argument("--write_report", metavar="FILE",
                   help="Write markdown table to FILE (use with --summarize)")
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    if args.summarize:
        print_summary()
        if args.write_report:
            write_report(args.write_report)
        return

    if not args.model:
        p.error("--model is required unless --summarize is used")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if args.model == "ensemble":
        dino_ckpt = best_ckpt(CKPT_DIRS["dino_large"])
        eva_ckpt  = best_ckpt(CKPT_DIRS["eva"])
        print(f"Loading DINO large from {dino_ckpt}...", flush=True)
        dino_large = DinoV3_1.load_from_checkpoint(dino_ckpt).to(device)
        dino_tf = build_transform(dino_large.hparams.backbone_name, 224)
        print(f"Loading EVA from {eva_ckpt}...", flush=True)
        eva = EvaViT.load_from_checkpoint(eva_ckpt).to(device)
        eva_tf = build_transform(eva.hparams.backbone_name, 448)
        print("Running ensemble inference on JRAIGS train...", flush=True)
        metrics = run_ensemble(
            dino_large, eva,
            build_loader(dino_tf, args.batch_size, args.num_workers, args.data_dir),
            build_loader(eva_tf,  args.batch_size, args.num_workers, args.data_dir),
            device,
        )

    elif args.model == "dino_large":
        ckpt = best_ckpt(CKPT_DIRS["dino_large"])
        print(f"Loading DINO large from {ckpt}...", flush=True)
        model = DinoV3_1.load_from_checkpoint(ckpt).to(device)
        tf = build_transform(model.hparams.backbone_name, 224)
        print("Running inference on JRAIGS train...", flush=True)
        metrics = run_eval(model, build_loader(tf, args.batch_size, args.num_workers, args.data_dir), device)

    elif args.model == "eva":
        ckpt = best_ckpt(CKPT_DIRS["eva"])
        print(f"Loading EVA from {ckpt}...", flush=True)
        model = EvaViT.load_from_checkpoint(ckpt).to(device)
        tf = build_transform(model.hparams.backbone_name, 448)
        print("Running inference on JRAIGS train...", flush=True)
        metrics = run_eval(model, build_loader(tf, args.batch_size, args.num_workers, args.data_dir), device)

    elif args.model == "dino_small":
        ckpt = best_ckpt(CKPT_DIRS["dino_small"])
        print(f"Loading DINO small from {ckpt}...", flush=True)
        model = DinoV3_1.load_from_checkpoint(ckpt).to(device)
        tf = build_transform(model.hparams.backbone_name, 224)
        print("Running inference on JRAIGS train...", flush=True)
        metrics = run_eval(model, build_loader(tf, args.batch_size, args.num_workers, args.data_dir), device)

    elif args.model in ("distil_single", "distil_dual"):
        ckpt = best_ckpt(CKPT_DIRS[args.model])
        print(f"Loading {args.model} from {ckpt}...", flush=True)
        cls = StudentGlaucomaDistilled if args.model == "distil_single" else StudentGlaucomaDualDistilled
        model = cls.load_from_checkpoint(ckpt, strict=False).to(device)
        tf = build_transform(model.hparams.backbone_name, model.hparams.img_size)
        print("Running inference on JRAIGS train...", flush=True)
        metrics = run_eval(model, build_loader(tf, args.batch_size, args.num_workers, args.data_dir), device)

    print(f"\nResults for {MODEL_NAMES[args.model]}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    save_result(args.model, metrics)


if __name__ == "__main__":
    main()
