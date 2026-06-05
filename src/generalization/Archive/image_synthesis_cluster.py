"""
Image synthesis + cluster distance experiment.

Existing cluster_gen results (no synthesis, RIMONE AUC):
  Cluster A (JRAIGS, far)       : 0.751
  Cluster B (ORIGA+LAG, far)    : 0.715
  Cluster C (ACRIMA+Harvard, close): 0.722
  A+B (no close)                : 0.701
  A+B+C (all)                   : 0.798

Question: does Reinhard-LAB or HistMatch synthesis help more when the
source domain is far from RIMONE (A or B) vs close (C)?
And does combining multiple far clusters (A+B) + synthesis beat A+B alone?

For each (source_combo, synthesis_method):
  1. Apply synthesis transform to source images → RIMONE-style training set
  2. Fine-tune MobileNetV3-small (backbone unfrozen) on synthetic images
  3. Evaluate on RIMONE test set
  4. Compare against no-synthesis baseline (=cluster_gen results)
"""

from __future__ import annotations

import copy
import json
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
from src.datasets.JRAIGS import JRAIGSDataset
from src.datasets.ORIGA import ORIGADataset
from src.datasets.LAG import LAGDataset
from src.datasets.RIMONE import RIMONEDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 42
N_PER_DS     = 200   # images sampled per source dataset
N_STYLE_REF  = 150   # RIMONE images as style reference
N_EVAL       = 200   # RIMONE images for evaluation
BACKBONE     = "mobilenetv3_small_100"
IMG_SIZE     = 224
BATCH_SIZE   = 16
N_EPOCHS     = 15
LR           = 3e-4
WEIGHT_DECAY = 1e-3
DATA_DIR     = "data/datasets"

# cluster_gen baseline AUCs on RIMONE (no synthesis, different sampling — for reference only)
BASELINE_AUCS = {
    "C only (close)":   0.722,
    "A only (far)":     0.751,
    "B only (far)":     0.715,
    "A+B (far only)":   0.701,
    "A+B+C (all)":      0.798,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_numpy_uint8(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)
    if isinstance(img, torch.Tensor):
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.asarray(img)
    return np.clip(arr, 0, 255).astype(np.uint8)


def sample_dataset(ds, n, seed=SEED, balanced=False):
    """Sample up to n images. If balanced=True, sample n//2 pos and n//2 neg."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g).tolist()

    if not balanced:
        images, labels = [], []
        for i in idx:
            if len(images) >= n:
                break
            item = ds[i]
            lbl = int(item["label"].item() if hasattr(item["label"], "item") else item["label"])
            if lbl not in (0, 1):
                continue
            images.append(to_numpy_uint8(item["image"]))
            labels.append(lbl)
        return images, np.array(labels, dtype=np.int32)

    # balanced: equal pos/neg up to n//2 each
    target = n // 2
    pos_imgs, neg_imgs = [], []
    for i in idx:
        if len(pos_imgs) >= target and len(neg_imgs) >= target:
            break
        item = ds[i]
        lbl = int(item["label"].item() if hasattr(item["label"], "item") else item["label"])
        if lbl == 1 and len(pos_imgs) < target:
            pos_imgs.append(to_numpy_uint8(item["image"]))
        elif lbl == 0 and len(neg_imgs) < target:
            neg_imgs.append(to_numpy_uint8(item["image"]))
    images = pos_imgs + neg_imgs
    labels = [1] * len(pos_imgs) + [0] * len(neg_imgs)
    return images, np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Image transforms
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


TRANSFORMS = {
    "Baseline":     None,
    "Reinhard-LAB": reinhard_lab,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynthDataset(Dataset):
    def __init__(self, images: list, labels: np.ndarray,
                 style_imgs: list, transform_fn, timm_tf, seed=SEED):
        self.images = images
        self.labels = labels
        self.style  = style_imgs
        self.fn     = transform_fn
        self.tf     = timm_tf
        self._rng   = np.random.default_rng(seed)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        src = self.images[idx]
        sty = self.style[int(self._rng.integers(len(self.style)))]
        h, w = src.shape[:2]
        sty_r = cv2.resize(sty, (w, h), interpolation=cv2.INTER_LINEAR)
        try:
            out = self.fn(src, sty_r)
        except Exception:
            out = src
        return self.tf(Image.fromarray(out.astype(np.uint8))), int(self.labels[idx])


class PlainDataset(Dataset):
    def __init__(self, images, labels, timm_tf):
        self.images = images
        self.labels = labels
        self.tf     = timm_tf

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.tf(Image.fromarray(self.images[idx])), int(self.labels[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=2)
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)
    tf_eval  = timm.data.create_transform(**data_cfg, is_training=False)
    tf_train = timm.data.create_transform(**data_cfg, is_training=True)
    return model, tf_eval, tf_train


def train_and_eval(model_init, train_ds, eval_ds, device):
    model = copy.deepcopy(model_init).to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=2, drop_last=False)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    y = np.array([train_ds.labels[i] for i in range(len(train_ds))])
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
            print(f"        epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/len(dl):.4f}", flush=True)

    model.eval()
    dl_eval = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, ys in dl_eval:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(np.array(ys))

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    acc  = float((preds == labels).mean())
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    bacc = (sens + spec) / 2          # balanced accuracy — robust à l'imbalance RIMONE
    return dict(auc=auc, acc=acc, bacc=float(bacc),
                sens=float(sens), spec=float(spec))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model_init, tf_eval, tf_train = build_model()
    print(f"Backbone: {BACKBONE}  params={sum(p.numel() for p in model_init.parameters()):,}", flush=True)

    # --- Load datasets ---
    print("\nLoading datasets …", flush=True)
    ds_acrima  = ACRIMADataset(data_dir=DATA_DIR)
    ds_harvard = HarvardGlaucomaDataset(data_dir=DATA_DIR)
    ds_jraigs  = JRAIGSDataset(data_dir=DATA_DIR)
    ds_origa   = ORIGADataset(data_dir=DATA_DIR)
    ds_lag     = LAGDataset(data_dir=DATA_DIR)
    ds_rimone  = RIMONEDataset(data_dir=DATA_DIR, split="train")

    imgs_acrima,  y_acrima  = sample_dataset(ds_acrima,  N_PER_DS,    seed=SEED,   balanced=True)
    imgs_harvard, y_harvard = sample_dataset(ds_harvard, N_PER_DS,    seed=SEED+1, balanced=True)
    imgs_jraigs,  y_jraigs  = sample_dataset(ds_jraigs,  N_PER_DS,    seed=SEED+2, balanced=True)
    imgs_origa,   y_origa   = sample_dataset(ds_origa,   N_PER_DS,    seed=SEED+3, balanced=True)
    imgs_lag,     y_lag     = sample_dataset(ds_lag,     N_PER_DS,    seed=SEED+4, balanced=True)
    imgs_style,   _         = sample_dataset(ds_rimone,  N_STYLE_REF, seed=SEED+5)
    imgs_rimone,  y_rimone  = sample_dataset(ds_rimone,  N_EVAL,      seed=SEED+6)

    # Source combinations
    source_combos = {
        "C only (close)":  (imgs_acrima + imgs_harvard,
                            np.concatenate([y_acrima, y_harvard])),
        "A only (far)":    (imgs_jraigs, y_jraigs),
        "B only (far)":    (imgs_origa + imgs_lag,
                            np.concatenate([y_origa, y_lag])),
        "A+B (far only)":  (imgs_jraigs + imgs_origa + imgs_lag,
                            np.concatenate([y_jraigs, y_origa, y_lag])),
        "A+B+C (all)":     (imgs_jraigs + imgs_origa + imgs_lag + imgs_acrima + imgs_harvard,
                            np.concatenate([y_jraigs, y_origa, y_lag, y_acrima, y_harvard])),
    }

    for name, (imgs, y) in source_combos.items():
        print(f"  {name}: {len(imgs)} imgs  pos={y.sum()}  neg={(y==0).sum()}", flush=True)
    print(f"  Style ref RIMONE: {len(imgs_style)} imgs", flush=True)
    print(f"  Eval  RIMONE    : {len(imgs_rimone)} imgs  pos={y_rimone.sum()}  neg={(y_rimone==0).sum()}", flush=True)

    eval_ds = PlainDataset(imgs_rimone, y_rimone, tf_eval)

    # --- Run experiments ---
    results = {}
    w = 80

    print(f"\n{'='*w}", flush=True)
    print("EXPERIMENT — synthesis × cluster source → RIMONE generalization", flush=True)
    print(f"{'='*w}", flush=True)

    for combo_name, (imgs_src, y_src) in source_combos.items():
        results[combo_name] = {}

        for synth_name, fn in TRANSFORMS.items():
            key = f"{combo_name} + {synth_name}"
            print(f"\n[{key}]", flush=True)
            t0 = time.time()

            if fn is None:
                train_ds = PlainDataset(imgs_src, y_src, tf_train)
            else:
                train_ds = SynthDataset(imgs_src, y_src, imgs_style, fn, tf_train, seed=SEED)
            set_seed(SEED)
            m = train_and_eval(model_init, train_ds, eval_ds, device)
            m["time"] = time.time() - t0
            results[combo_name][synth_name] = m

            baseline = BASELINE_AUCS.get(combo_name, float("nan"))
            delta = m["auc"] - baseline
            sign = "▲" if delta > 0 else "▼"
            print(f"  AUC={m['auc']:.4f} ({sign}{abs(delta):.4f} vs no-synth {baseline:.3f})"
                  f"  bAcc={m['bacc']:.4f}  Sens={m['sens']:.4f}  Spec={m['spec']:.4f}"
                  f"  ({m['time']:.0f}s)", flush=True)

    # --- Summary table ---
    print(f"\n{'='*w}", flush=True)
    print("RÉSULTATS — AUC / bAcc sur RIMONE  (balanced sampling, Reinhard-LAB vs Baseline)", flush=True)
    print(f"{'='*w}", flush=True)
    hdr = (f"{'Source':<22} {'AUC base':>9} {'AUC synth':>10} {'ΔAUC':>6}"
           f" {'bAcc base':>10} {'bAcc synth':>11} {'ΔbAcc':>7}")
    print(hdr, flush=True)
    print("-" * w, flush=True)

    for combo_name in source_combos:
        b  = results[combo_name]["Baseline"]
        r  = results[combo_name]["Reinhard-LAB"]
        da = r["auc"]  - b["auc"]
        db = r["bacc"] - b["bacc"]
        sa = ("▲" if da > 0 else "▼") + f"{abs(da):.3f}"
        sb = ("▲" if db > 0 else "▼") + f"{abs(db):.3f}"
        print(f"  {combo_name:<20} {b['auc']:>9.4f} {r['auc']:>10.4f} {sa:>6}"
              f" {b['bacc']:>10.4f} {r['bacc']:>11.4f} {sb:>7}", flush=True)

    print(f"{'='*w}", flush=True)

    # Save results
    out_path = Path("figures/image_synthesis_cluster/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
