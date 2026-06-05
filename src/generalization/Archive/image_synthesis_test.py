"""
Image-level synthesis → train on synthetic, test on RIMONE (unseen).

For each synthesis method:
  1. Apply pixel-level transform to ACRIMA+Harvard → synthetic training set
  2. Fine-tune a classifier (backbone fully unfrozen) on synthetic images
  3. Evaluate on real RIMONE images (never seen during training)

Methods:
  Baseline, HistMatch, Reinhard-LAB, FDA β=0.01/0.05/0.1,
  PixelAdaIN, CLAHE, FDA+Reinhard, Hist+CLAHE

Metric: AUC (primary), Accuracy, Sensitivity, Specificity
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.datasets import ACRIMADataset, HarvardGlaucomaDataset
from src.datasets.RIMONE import RIMONEDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 42
N_SOURCE     = 200   # images per source dataset (train)
N_STYLE_REF  = 150   # RIMONE images used as style reference (not for eval)
N_EVAL       = 200   # RIMONE images used for evaluation
BACKBONE     = "mobilenetv3_small_100"   # lightweight, fast to fine-tune
IMG_SIZE     = 224
BATCH_SIZE   = 16
N_EPOCHS     = 15
LR           = 3e-4
WEIGHT_DECAY = 1e-3
DATA_DIR     = "data/datasets"

rng_global = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_numpy_uint8(img) -> np.ndarray:
    """Accept PIL Image, numpy array, or tensor → HxWxC uint8 numpy."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)
    if isinstance(img, torch.Tensor):
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def sample_dataset(ds, n, seed=SEED):
    """Sample up to n images with binary labels (0 or 1); skips other values."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g).tolist()
    images, labels = [], []
    for i in idx:
        if len(images) >= n:
            break
        item = ds[i]
        lbl = int(item["label"].item() if hasattr(item["label"], "item") else item["label"])
        if lbl not in (0, 1):
            continue
        img = to_numpy_uint8(item["image"])
        images.append(img)
        labels.append(lbl)
    return images, np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Image transforms (pixel-level, unsupervised)
# ---------------------------------------------------------------------------

def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(src)
    for c in range(3):
        s, r = src[:, :, c].ravel(), ref[:, :, c].ravel()
        s_vals, s_counts = np.unique(s, return_counts=True)
        r_vals, r_counts = np.unique(r, return_counts=True)
        s_cdf = np.cumsum(s_counts).astype(float) / s.size
        r_cdf = np.cumsum(r_counts).astype(float) / r.size
        lut = np.interp(s_cdf, r_cdf, r_vals).astype(np.uint8)
        full_lut = np.zeros(256, dtype=np.uint8)
        full_lut[s_vals] = lut
        out[:, :, c] = full_lut[src[:, :, c]]
    return out


def reinhard_lab(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_f = src.astype(np.float32) / 255.0
    ref_f = ref.astype(np.float32) / 255.0
    src_lab = cv2.cvtColor(src_f, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(ref_f, cv2.COLOR_RGB2LAB)
    out_lab = np.empty_like(src_lab)
    for c in range(3):
        mu_s, sd_s = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        mu_r, sd_r = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std() + 1e-6
        out_lab[:, :, c] = (src_lab[:, :, c] - mu_s) / sd_s * sd_r + mu_r
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return to_uint8(np.clip(out_rgb * 255, 0, 255))


def fda(src: np.ndarray, ref: np.ndarray, beta: float = 0.01) -> np.ndarray:
    h, w = src.shape[:2]
    b = max(1, int(min(h, w) * beta))
    out = np.empty_like(src)
    for c in range(3):
        fs = np.fft.fft2(src[:, :, c].astype(np.float32))
        fr = np.fft.fft2(ref[:, :, c].astype(np.float32))
        fs_shift = np.fft.fftshift(fs)
        fr_shift = np.fft.fftshift(fr)
        amp_s = np.abs(fs_shift)
        amp_r = np.abs(fr_shift)
        pha_s = np.angle(fs_shift)
        cy, cx = h // 2, w // 2
        amp_s[cy-b:cy+b, cx-b:cx+b] = amp_r[cy-b:cy+b, cx-b:cx+b]
        fs_new = amp_s * np.exp(1j * pha_s)
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(fs_new)))
    return to_uint8(out)


def pixel_adain(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(src)
    for c in range(3):
        s = src[:, :, c].astype(np.float32)
        r = ref[:, :, c].astype(np.float32)
        out[:, :, c] = (s - s.mean()) / (s.std() + 1e-6) * r.std() + r.mean()
    return to_uint8(out)


def clahe_norm(src: np.ndarray, _ref=None) -> np.ndarray:
    src_f = src.astype(np.float32) / 255.0
    lab = cv2.cvtColor(src_f, cv2.COLOR_RGB2LAB)
    L = (lab[:, :, 0] / 100.0 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    lab[:, :, 0] = L_eq.astype(np.float32) / 255.0 * 100.0
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return to_uint8(np.clip(rgb * 255, 0, 255))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynthDataset(Dataset):
    """Applies a synthesis transform at load time; style ref sampled randomly."""

    def __init__(self, images: list[np.ndarray], labels: np.ndarray,
                 style_imgs: list[np.ndarray] | None,
                 transform_fn,
                 timm_tf,
                 seed: int = SEED):
        self.images = images
        self.labels = labels
        self.style  = style_imgs
        self.fn     = transform_fn
        self.tf     = timm_tf
        self._rng   = np.random.default_rng(seed)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        src = self.images[idx]  # HxWxC uint8 numpy

        if self.style is not None:
            sty = self.style[int(self._rng.integers(len(self.style)))]
            h, w = src.shape[:2]
            sty_r = cv2.resize(sty, (w, h), interpolation=cv2.INTER_LINEAR)
            try:
                out = self.fn(src, sty_r)
            except Exception:
                out = src
        else:
            out = src

        pil = Image.fromarray(out.astype(np.uint8))
        return self.tf(pil), int(self.labels[idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2) -> tuple[nn.Module, object]:
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=num_classes)
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)
    timm_tf = timm.data.create_transform(**data_cfg, is_training=False)
    train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    return model, timm_tf, train_tf


# ---------------------------------------------------------------------------
# Training (one phase, backbone unfrozen)
# ---------------------------------------------------------------------------

def train_one_method(model_init: nn.Module, train_ds: Dataset,
                     device: torch.device) -> nn.Module:
    model = copy.deepcopy(model_init).to(device)
    # all params unfrozen
    for p in model.parameters():
        p.requires_grad_(True)

    dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, drop_last=False)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # class weights to handle imbalance
    y = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    n_pos, n_neg = y.sum(), (y == 0).sum()
    w = torch.tensor([n_pos / len(y), n_neg / len(y)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for imgs, ys in dl:
            ys = ys.clone().detach().long() if isinstance(ys, torch.Tensor) else torch.tensor(ys).long()
            imgs, ys = imgs.to(device), ys.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/len(dl):.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation on RIMONE
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_rimone(model: nn.Module, eval_ds: Dataset,
                       device: torch.device) -> dict:
    dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    for imgs, ys in dl:
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(np.array(ys))

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    acc  = (preds == labels).mean()
    tp   = ((preds == 1) & (labels == 1)).sum()
    tn   = ((preds == 0) & (labels == 0)).sum()
    fp   = ((preds == 1) & (labels == 0)).sum()
    fn   = ((preds == 0) & (labels == 1)).sum()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return dict(auc=auc, acc=acc, sens=sens, spec=spec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {BACKBONE}  epochs={N_EPOCHS}  lr={LR}")

    # Build model template once (weights shared as starting point)
    model_init, timm_tf, train_tf = build_model(num_classes=2)
    print(f"  Parameters: {sum(p.numel() for p in model_init.parameters()):,}")

    # --- Load raw images ---
    print("\nLoading datasets …")
    ds_acrima  = ACRIMADataset(data_dir=DATA_DIR)
    ds_harvard = HarvardGlaucomaDataset(data_dir=DATA_DIR)
    ds_rimone  = RIMONEDataset(data_dir=DATA_DIR, split="train")

    imgs_acrima,  y_acrima  = sample_dataset(ds_acrima,  N_SOURCE,    seed=SEED)
    imgs_harvard, y_harvard = sample_dataset(ds_harvard, N_SOURCE,    seed=SEED+1)
    imgs_style,   _         = sample_dataset(ds_rimone,  N_STYLE_REF, seed=SEED+2)
    imgs_rimone,  y_rimone  = sample_dataset(ds_rimone,  N_EVAL,      seed=SEED+3)

    imgs_src = imgs_acrima + imgs_harvard
    y_src    = np.concatenate([y_acrima, y_harvard])

    print(f"  Train source  (ACRIMA+Harvard): {len(imgs_src)} images  "
          f"pos={y_src.sum()}  neg={(y_src==0).sum()}")
    print(f"  Style ref   RIMONE           : {len(imgs_style)} images")
    print(f"  Test target RIMONE           : {len(imgs_rimone)} images  "
          f"pos={y_rimone.sum()}  neg={(y_rimone==0).sum()}")

    # Eval dataset (fixed, no transform)
    eval_ds = SynthDataset(imgs_rimone, y_rimone, None,
                           lambda s, r: s, timm_tf, seed=SEED)

    # --- Methods ---
    methods = {
        "Baseline":      (lambda s, r: s,                                 False),
        "HistMatch":     (histogram_match,                                True),
        "Reinhard-LAB":  (reinhard_lab,                                   True),
        "FDA β=0.01":    (lambda s, r: fda(s, r, 0.01),                  True),
        "FDA β=0.05":    (lambda s, r: fda(s, r, 0.05),                  True),
        "FDA β=0.1":     (lambda s, r: fda(s, r, 0.1),                   True),
        "PixelAdaIN":    (pixel_adain,                                    True),
        "CLAHE":         (clahe_norm,                                     False),
        "FDA+Reinhard":  (lambda s, r: reinhard_lab(fda(s, r, 0.05), r), True),
        "Hist+CLAHE":    (lambda s, r: clahe_norm(histogram_match(s, r)), True),
    }

    results = []

    print(f"\n{'='*70}")
    print("EXPERIMENT — train on synthetic, evaluate on unseen RIMONE")
    print(f"{'='*70}")

    for name, (fn, needs_style) in methods.items():
        print(f"\n[{name}]  fine-tuning …")
        t0 = time.time()

        style = imgs_style if needs_style else None
        train_ds = SynthDataset(imgs_src, y_src, style, fn, train_tf, seed=SEED)

        set_seed(SEED)
        model = train_one_method(model_init, train_ds, device)
        metrics = evaluate_on_rimone(model, eval_ds, device)
        elapsed = time.time() - t0

        metrics["name"] = name
        metrics["time"] = elapsed
        results.append(metrics)
        print(f"      → AUC={metrics['auc']:.4f}  Acc={metrics['acc']:.4f}"
              f"  Sens={metrics['sens']:.4f}  Spec={metrics['spec']:.4f}"
              f"  ({elapsed:.0f}s)")

    # --- Summary table ---
    w = 70
    print(f"\n{'='*w}")
    print(f"RÉSULTATS — généralisation zéro-shot vers RIMONE")
    print(f"{'='*w}")
    hdr = f"{'Méthode':<18} {'AUC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'t(s)':>6}"
    print(hdr)
    print("-" * w)
    for r in sorted(results, key=lambda x: -x["auc"]):
        print(f"  {r['name']:<16} {r['auc']:>7.4f} {r['acc']:>7.4f}"
              f" {r['sens']:>7.4f} {r['spec']:>7.4f} {r['time']:>6.0f}")
    print(f"{'='*w}")

    best = max(results, key=lambda x: x["auc"])
    print(f"\nMeilleure méthode : {best['name']}  (AUC={best['auc']:.4f})")


if __name__ == "__main__":
    main()
