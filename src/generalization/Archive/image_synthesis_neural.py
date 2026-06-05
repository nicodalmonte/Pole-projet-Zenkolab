"""
Neural AdaIN style transfer for domain adaptation to RIMONE.

Architecture:
  - Encoder : VGG19 up to relu4_1 (pretrained ImageNet, frozen)
  - AdaIN   : match feature mean/std of content to style at relu4_1
  - Decoder : learned upsampling CNN (trained here on reconstruction)

Key difference vs pixel-level methods (Reinhard, pixel AdaIN):
  Style matching happens in deep feature space (512 channels, H/8×W/8),
  capturing mid-level texture statistics, not just global colour.

Pipeline:
  1. Train decoder once on all available images (reconstruction objective:
     L1 pixel + VGG perceptual content loss at relu3_1 and relu4_1).
  2. For each source combo: transform all source images with neural AdaIN
     using a random RIMONE image as style reference.
  3. Train MobileNetV3-small classifier (backbone unfrozen) on transformed images.
  4. Evaluate on held-out RIMONE test set.

Comparison point: Baseline and Reinhard-LAB results loaded from
  figures/image_synthesis_cluster/results.json  (same sampling, same eval set).
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
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.datasets import ACRIMADataset, HarvardGlaucomaDataset
from src.datasets.JRAIGS import JRAIGSDataset
from src.datasets.LAG import LAGDataset
from src.datasets.ORIGA import ORIGADataset
from src.datasets.RIMONE import RIMONEDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED            = 42
N_PER_DS        = 200
N_STYLE_REF     = 150
N_EVAL          = 200
BACKBONE_CLS    = "mobilenetv3_small_100"
IMG_SIZE        = 224
BATCH_SIZE      = 16
N_EPOCHS_CLS    = 15
LR_CLS          = 3e-4
WEIGHT_DECAY    = 1e-3
N_EPOCHS_DEC    = 20
LR_DEC          = 1e-3
DECODER_CKPT    = Path("checkpoints/adain_decoder.pt")
DATA_DIR        = "data/datasets"

# ImageNet normalisation used by VGG encoder
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
VGG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def to_numpy_uint8(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)
    if isinstance(img, torch.Tensor):
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip(np.asarray(img), 0, 255).astype(np.uint8)


def sample_dataset(ds, n, seed=SEED, balanced=False):
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

    target = n // 2
    pos, neg = [], []
    for i in idx:
        if len(pos) >= target and len(neg) >= target:
            break
        item = ds[i]
        lbl = int(item["label"].item() if hasattr(item["label"], "item") else item["label"])
        img = to_numpy_uint8(item["image"])
        if lbl == 1 and len(pos) < target:
            pos.append(img)
        elif lbl == 0 and len(neg) < target:
            neg.append(img)
    return pos + neg, np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)


def imgs_to_tensor(images: list[np.ndarray], size: int = IMG_SIZE) -> torch.Tensor:
    """HxWxC uint8 list → NCHW float [0,1] tensor, resized to size×size."""
    out = []
    for img in images:
        img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(out)   # (N, 3, H, W)


# ---------------------------------------------------------------------------
# VGG Encoder — frozen, up to relu4_1
# ---------------------------------------------------------------------------

class VGGEncoder(nn.Module):
    """VGG19 split into 4 slices ending at relu1_1/2_1/3_1/4_1."""

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])    # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])   # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])  # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21]) # relu4_1
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, return_all: bool = False):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return (h1, h2, h3, h4) if return_all else h4


def vgg_norm(x: torch.Tensor, device) -> torch.Tensor:
    mean = VGG_MEAN.to(device)
    std  = VGG_STD.to(device)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# AdaIN
# ---------------------------------------------------------------------------

def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    B, C = content_feat.shape[:2]
    cf = content_feat.view(B, C, -1)
    sf = style_feat.view(B, C, -1)
    mu_c = cf.mean(dim=2, keepdim=True).unsqueeze(-1)
    sd_c = cf.std(dim=2, keepdim=True).unsqueeze(-1) + 1e-5
    mu_s = sf.mean(dim=2, keepdim=True).unsqueeze(-1)
    sd_s = sf.std(dim=2, keepdim=True).unsqueeze(-1) + 1e-5
    return (content_feat - mu_c) / sd_c * sd_s + mu_s


# ---------------------------------------------------------------------------
# Decoder — relu4_1 (512, H/8, W/8) → RGB
# ---------------------------------------------------------------------------

def _rc(in_c, out_c, k=3):
    """ReflectionPad + Conv + ReLU block."""
    return nn.Sequential(
        nn.ReflectionPad2d(k // 2),
        nn.Conv2d(in_c, out_c, k),
        nn.ReLU(inplace=True),
    )


class AdaINDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _rc(512, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _rc(256, 256),
            _rc(256, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _rc(128, 128),
            _rc(128, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _rc(64, 64),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ---------------------------------------------------------------------------
# Decoder training
# ---------------------------------------------------------------------------

def perceptual_content_loss(encoder: VGGEncoder,
                             recon: torch.Tensor,
                             target: torch.Tensor,
                             device) -> torch.Tensor:
    """VGG content loss at relu3_1 and relu4_1."""
    _, _, f_r3, f_r4 = encoder(vgg_norm(recon.clamp(0, 1), device), return_all=True)
    with torch.no_grad():
        _, _, t_r3, t_r4 = encoder(vgg_norm(target, device), return_all=True)
    return nn.functional.l1_loss(f_r3, t_r3) + nn.functional.l1_loss(f_r4, t_r4)


def train_decoder(encoder: VGGEncoder,
                  all_images: list[np.ndarray],
                  device: torch.device,
                  save_path: Path) -> AdaINDecoder:
    print(f"\n--- Training AdaIN decoder on {len(all_images)} images "
          f"({N_EPOCHS_DEC} epochs) ---", flush=True)
    decoder = AdaINDecoder().to(device)
    optimizer = Adam(decoder.parameters(), lr=LR_DEC)

    imgs_t = imgs_to_tensor(all_images).to(device)   # (N, 3, H, W) [0,1]
    dataset = TensorDataset(imgs_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    for epoch in range(N_EPOCHS_DEC):
        total = 0.0
        for (batch,) in loader:
            batch_norm = vgg_norm(batch, device)
            with torch.no_grad():
                feat = encoder(batch_norm)
            recon = decoder(feat)
            # L1 pixel + VGG perceptual content loss
            loss = nn.functional.l1_loss(recon, batch) + \
                   0.1 * perceptual_content_loss(encoder, recon, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  decoder epoch {epoch+1}/{N_EPOCHS_DEC}  loss={total/len(loader):.4f}",
                  flush=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"  Decoder saved → {save_path}", flush=True)
    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Neural AdaIN transform: source images → RIMONE-styled images
# ---------------------------------------------------------------------------

@torch.no_grad()
def neural_adain_transform(source_imgs: list[np.ndarray],
                            style_imgs: list[np.ndarray],
                            encoder: VGGEncoder,
                            decoder: AdaINDecoder,
                            device: torch.device,
                            seed: int = SEED) -> list[np.ndarray]:
    """Return list of uint8 numpy images styled to match random RIMONE refs."""
    rng = np.random.default_rng(seed)
    encoder.eval()
    decoder.eval()

    style_t = imgs_to_tensor(style_imgs).to(device)   # (M, 3, H, W)
    result   = []

    for i in range(0, len(source_imgs), BATCH_SIZE):
        batch_np = source_imgs[i:i + BATCH_SIZE]
        content_t = imgs_to_tensor(batch_np).to(device)

        # sample one random style per content image
        sty_idx = rng.integers(len(style_imgs), size=len(batch_np))
        sty_batch = style_t[sty_idx]

        c_feat = encoder(vgg_norm(content_t, device))
        s_feat = encoder(vgg_norm(sty_batch, device))

        transferred_feat = adain(c_feat, s_feat)
        recon = decoder(transferred_feat).clamp(0, 1)  # (B, 3, H, W)

        for img_t in recon:
            arr = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            result.append(arr)

    return result


# ---------------------------------------------------------------------------
# Classifier datasets
# ---------------------------------------------------------------------------

class PlainDataset(Dataset):
    def __init__(self, images, labels, timm_tf):
        self.images = images
        self.labels = np.asarray(labels)
        self.tf = timm_tf

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return self.tf(Image.fromarray(img)), int(self.labels[idx])


# ---------------------------------------------------------------------------
# Classifier training & evaluation
# ---------------------------------------------------------------------------

def build_classifier():
    model = timm.create_model(BACKBONE_CLS, pretrained=True, num_classes=2)
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
    y  = train_ds.labels
    w  = torch.tensor([y.sum() / len(y), (y == 0).sum() / len(y)],
                      dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = AdamW(model.parameters(), lr=LR_CLS, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(N_EPOCHS_CLS):
        total = 0.0
        for imgs, ys in dl:
            ys = ys.clone().detach().long() if isinstance(ys, torch.Tensor) \
                 else torch.tensor(ys).long()
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), ys.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"        epoch {epoch+1}/{N_EPOCHS_CLS}  loss={total/len(dl):.4f}",
                  flush=True)

    model.eval()
    dl_e = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, ys in dl_e:
            p = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            probs_all.append(p)
            labels_all.append(np.asarray(ys))

    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds  = (probs >= 0.5).astype(int)
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return dict(auc=float(auc), acc=float((preds == labels).mean()),
                bacc=float((sens + spec) / 2),
                sens=float(sens), spec=float(spec))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # --- Load classifier backbone template ---
    model_init, tf_eval, tf_train = build_classifier()
    print(f"Classifier: {BACKBONE_CLS}  "
          f"params={sum(p.numel() for p in model_init.parameters()):,}", flush=True)

    # --- Load VGG encoder ---
    print("Loading VGG19 encoder …", flush=True)
    encoder = VGGEncoder().to(device).eval()

    # --- Load datasets ---
    print("\nLoading datasets …", flush=True)
    ds_acrima  = ACRIMADataset(data_dir=DATA_DIR)
    ds_harvard = HarvardGlaucomaDataset(data_dir=DATA_DIR)
    ds_jraigs  = JRAIGSDataset(data_dir=DATA_DIR)
    ds_origa   = ORIGADataset(data_dir=DATA_DIR)
    ds_lag     = LAGDataset(data_dir=DATA_DIR)
    ds_rimone  = RIMONEDataset(data_dir=DATA_DIR, split="train")

    imgs_acrima,  y_acrima  = sample_dataset(ds_acrima,  N_PER_DS, seed=SEED,   balanced=True)
    imgs_harvard, y_harvard = sample_dataset(ds_harvard, N_PER_DS, seed=SEED+1, balanced=True)
    imgs_jraigs,  y_jraigs  = sample_dataset(ds_jraigs,  N_PER_DS, seed=SEED+2, balanced=True)
    imgs_origa,   y_origa   = sample_dataset(ds_origa,   N_PER_DS, seed=SEED+3, balanced=True)
    imgs_lag,     y_lag     = sample_dataset(ds_lag,     N_PER_DS, seed=SEED+4, balanced=True)
    imgs_style,   _         = sample_dataset(ds_rimone,  N_STYLE_REF, seed=SEED+5)
    imgs_rimone,  y_rimone  = sample_dataset(ds_rimone,  N_EVAL,   seed=SEED+6)

    source_combos = {
        "C only (close)": (imgs_acrima + imgs_harvard,
                           np.concatenate([y_acrima, y_harvard])),
        "A only (far)":   (imgs_jraigs, y_jraigs),
        "B only (far)":   (imgs_origa + imgs_lag,
                           np.concatenate([y_origa, y_lag])),
        "A+B (far only)": (imgs_jraigs + imgs_origa + imgs_lag,
                           np.concatenate([y_jraigs, y_origa, y_lag])),
        "A+B+C (all)":    (imgs_jraigs + imgs_origa + imgs_lag +
                           imgs_acrima + imgs_harvard,
                           np.concatenate([y_jraigs, y_origa, y_lag,
                                           y_acrima, y_harvard])),
    }
    for name, (imgs, y) in source_combos.items():
        print(f"  {name}: {len(imgs)} imgs  pos={y.sum()}  neg={(y==0).sum()}", flush=True)
    print(f"  Style ref RIMONE : {len(imgs_style)} imgs", flush=True)
    print(f"  Eval RIMONE      : {len(imgs_rimone)} imgs  "
          f"pos={y_rimone.sum()}  neg={(y_rimone==0).sum()}", flush=True)

    eval_ds = PlainDataset(imgs_rimone, y_rimone, tf_eval)

    # --- Train / load decoder ---
    if DECODER_CKPT.exists():
        print(f"\nLoading pre-trained decoder from {DECODER_CKPT} …", flush=True)
        decoder = AdaINDecoder().to(device)
        decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=device))
        decoder.eval()
    else:
        all_imgs_for_decoder = (imgs_acrima + imgs_harvard + imgs_jraigs +
                                imgs_origa + imgs_lag + imgs_style)
        decoder = train_decoder(encoder, all_imgs_for_decoder, device, DECODER_CKPT)

    # --- Pre-compute neural AdaIN transforms for each combo ---
    print("\nApplying neural AdaIN transforms …", flush=True)
    transformed_combos = {}
    for name, (imgs_src, y_src) in source_combos.items():
        print(f"  transforming {name} ({len(imgs_src)} imgs) …", flush=True)
        t0 = time.time()
        styled = neural_adain_transform(imgs_src, imgs_style, encoder, decoder,
                                        device, seed=SEED)
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
        transformed_combos[name] = (styled, y_src)

    # --- Classifier training & evaluation ---
    results = {}
    w = 82

    print(f"\n{'='*w}", flush=True)
    print("EXPERIMENT — Neural AdaIN style transfer → RIMONE generalization", flush=True)
    print(f"{'='*w}", flush=True)

    for combo_name, (imgs_styled, y_src) in transformed_combos.items():
        print(f"\n[{combo_name} + Neural AdaIN]", flush=True)
        t0 = time.time()
        train_ds = PlainDataset(imgs_styled, y_src, tf_train)
        set_seed(SEED)
        m = train_and_eval(model_init, train_ds, eval_ds, device)
        m["time"] = time.time() - t0
        results[combo_name] = m
        print(f"  AUC={m['auc']:.4f}  bAcc={m['bacc']:.4f}"
              f"  Sens={m['sens']:.4f}  Spec={m['spec']:.4f}"
              f"  ({m['time']:.0f}s)", flush=True)

    # --- Load previous results for comparison ---
    prev_path = Path("figures/image_synthesis_cluster/results.json")
    prev = {}
    if prev_path.exists():
        with open(prev_path) as f:
            raw = json.load(f)
        for k in source_combos:
            if k in raw:
                prev[k] = raw[k]

    # --- Summary table ---
    print(f"\n{'='*w}", flush=True)
    print("RÉSULTATS — AUC / bAcc sur RIMONE  (balanced sampling)", flush=True)
    print(f"{'='*w}", flush=True)
    hdr = (f"{'Source':<22} {'Baseline':>9} {'Reinhard':>9} {'NeuralAdaIN':>12}"
           f" {'ΔAUC':>6}  {'bAcc(base)':>11} {'bAcc(neural)':>13} {'ΔbAcc':>6}")
    print(hdr, flush=True)
    print("-" * w, flush=True)

    for combo_name in source_combos:
        n   = results[combo_name]
        b   = prev.get(combo_name, {}).get("Baseline",     {})
        r   = prev.get(combo_name, {}).get("Reinhard-LAB", {})
        auc_b = b.get("auc",  float("nan"))
        auc_r = r.get("auc",  float("nan"))
        auc_n = n["auc"]
        bac_b = b.get("bacc", float("nan"))
        bac_n = n["bacc"]
        da = auc_n - auc_b
        db = bac_n - bac_b
        sa = ("▲" if da > 0 else "▼") + f"{abs(da):.3f}"
        sb = ("▲" if db > 0 else "▼") + f"{abs(db):.3f}"
        print(f"  {combo_name:<20} {auc_b:>9.4f} {auc_r:>9.4f} {auc_n:>12.4f}"
              f" {sa:>6}  {bac_b:>11.4f} {bac_n:>13.4f} {sb:>6}", flush=True)
    print(f"{'='*w}", flush=True)

    # Save
    out_path = Path("figures/image_synthesis_neural/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
