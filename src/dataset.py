"""
Dataset classes for glaucoma classification.

Supported formats (auto-detected or explicit):
  ① LAG / REFUGE2 filename convention  → GlaucomaDataset
       g.XXXX.jpg / g0001.jpg  → 1 (glaucoma)
       ng.XXXX.jpg / n0001.jpg → 0 (normal)
       Im<id>_g_ACRIMA.jpg     → 1 (glaucoma)   ← ACRIMA
       Im<id>_ACRIMA.jpg       → 0 (normal)      ← ACRIMA

  ② Class-subfolder layout            → FolderGlaucomaDataset
       Glaucoma_Positive/ → 1  |  Glaucoma_Negative/ → 0
       (used by Fundus_Train_Val_Data)

  ③ CSV-labelled images               → CSVGlaucomaDataset
       CSV columns: Filename, Glaucoma (0/1)
       (used by ORIGA + glaucoma.csv)

MultiDirDataset concatenates any mix of the above.
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import pandas as pd


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_transforms(split: str) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# ① Label helper — LAG / REFUGE2 / ACRIMA filename conventions
# ---------------------------------------------------------------------------
def _label_from_name(name: str) -> int | None:
    """
    Returns 1 (glaucoma) / 0 (normal) from filename, or None to skip.
    Handles:  LAG (g./ng.), REFUGE2 (g0001/n0001), ACRIMA (_g_ACRIMA / _ACRIMA)
    """
    stem = Path(name).stem  # without extension
    # ACRIMA: Im<id>_g_ACRIMA  vs  Im<id>_ACRIMA
    if stem.startswith("Im") and "_ACRIMA" in stem:
        return 1 if "_g_ACRIMA" in stem else 0
    # LAG: ng. must be checked before g.
    if name.startswith("ng"):
        return 0
    if name.startswith("g"):
        return 1
    if name.startswith("n"):
        return 0
    return None


# ---------------------------------------------------------------------------
# Shared item loader
# ---------------------------------------------------------------------------
def _load_image(fpath: Path, transform) -> torch.Tensor:
    return transform(Image.open(fpath).convert("RGB"))


# ---------------------------------------------------------------------------
# ① GlaucomaDataset — filename-convention (LAG, REFUGE2, ACRIMA)
# ---------------------------------------------------------------------------
class GlaucomaDataset(Dataset):
    """
    Glaucoma dataset where labels are encoded in filenames.
    Supports LAG (g./ng.), REFUGE2 (g0001/n0001), ACRIMA (_g_ACRIMA).
    Images may be directly in root_dir or inside an images/ subdirectory.
    """

    def __init__(self, root_dir: str | Path, split: str = "train", transform=None):
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = transform or build_transforms(split)
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        img_dir = self.root_dir / "images" if (self.root_dir / "images").is_dir() \
                  else self.root_dir
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for fpath in sorted(img_dir.glob(ext)):
                label = _label_from_name(fpath.name)
                if label is not None:
                    self.samples.append((fpath, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No labelled images in {self.root_dir}. "
                "Expected filenames starting with g/n/ng or ACRIMA convention."
            )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        return {"image": _load_image(fpath, self.transform), "label": label, "path": str(fpath)}

    @property
    def class_weights(self) -> torch.Tensor:
        counts = [sum(1 for _, l in self.samples if l == i) for i in range(2)]
        total  = sum(counts)
        return torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)

    def __repr__(self):
        n_g = sum(1 for _, l in self.samples if l == 1)
        n_n = sum(1 for _, l in self.samples if l == 0)
        return f"GlaucomaDataset(split={self.split}, total={len(self)}, glaucoma={n_g}, normal={n_n}, src={self.root_dir.name})"


# ---------------------------------------------------------------------------
# ② FolderGlaucomaDataset — class-subfolder layout (Fundus dataset)
# ---------------------------------------------------------------------------
class FolderGlaucomaDataset(Dataset):
    """
    Layout:  root/Glaucoma_Positive/*.jpg  → label 1
             root/Glaucoma_Negative/*.jpg  → label 0
    Used by Fundus_Train_Val_Data (Train/ and Validation/ splits).
    """
    LABEL_MAP = {"Glaucoma_Positive": 1, "Glaucoma_Negative": 0}

    def __init__(self, root_dir: str | Path, split: str = "train", transform=None):
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = transform or build_transforms(split)
        self.samples: list[tuple[Path, int]] = []
        for subfolder, label in self.LABEL_MAP.items():
            folder = self.root_dir / subfolder
            if folder.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for fpath in sorted(folder.glob(ext)):
                        self.samples.append((fpath, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {self.root_dir}. "
                "Expected Glaucoma_Positive/ and Glaucoma_Negative/ subfolders."
            )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        return {"image": _load_image(fpath, self.transform), "label": label, "path": str(fpath)}

    @property
    def class_weights(self) -> torch.Tensor:
        counts = [sum(1 for _, l in self.samples if l == i) for i in range(2)]
        total  = sum(counts)
        return torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)

    def __repr__(self):
        n_g = sum(1 for _, l in self.samples if l == 1)
        n_n = sum(1 for _, l in self.samples if l == 0)
        return f"FolderGlaucomaDataset(split={self.split}, total={len(self)}, glaucoma={n_g}, normal={n_n}, src={self.root_dir.name})"


# ---------------------------------------------------------------------------
# ③ CSVGlaucomaDataset — CSV-labelled images (ORIGA)
# ---------------------------------------------------------------------------
class CSVGlaucomaDataset(Dataset):
    """
    Layout:  img_dir/*.jpg   labels from a CSV with columns [Filename, Glaucoma].
    Used by ORIGA + glaucoma.csv.
    """

    def __init__(self, img_dir: str | Path, csv_path: str | Path,
                 split: str = "train", transform=None):
        self.img_dir   = Path(img_dir)
        self.split     = split
        self.transform = transform or build_transforms(split)
        df = pd.read_csv(csv_path)
        self.samples: list[tuple[Path, int]] = []
        for _, row in df.iterrows():
            fpath = self.img_dir / row["Filename"]
            if fpath.exists():
                self.samples.append((fpath, int(row["Glaucoma"])))
        if not self.samples:
            raise FileNotFoundError(f"No images matched from CSV in {img_dir}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        return {"image": _load_image(fpath, self.transform), "label": label, "path": str(fpath)}

    @property
    def class_weights(self) -> torch.Tensor:
        counts = [sum(1 for _, l in self.samples if l == i) for i in range(2)]
        total  = sum(counts)
        return torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)

    def __repr__(self):
        n_g = sum(1 for _, l in self.samples if l == 1)
        n_n = sum(1 for _, l in self.samples if l == 0)
        return f"CSVGlaucomaDataset(split={self.split}, total={len(self)}, glaucoma={n_g}, normal={n_n}, src={self.img_dir.name})"


# ---------------------------------------------------------------------------
# MultiDirDataset — concatenates any mix of the above
# ---------------------------------------------------------------------------
class MultiDirDataset(Dataset):
    """
    Concatenates a list of already-built Dataset objects.
    Each item in `datasets` can be GlaucomaDataset, FolderGlaucomaDataset, or CSVGlaucomaDataset.
    """

    def __init__(self, datasets: list[Dataset], split: str = "train"):
        self.split     = split
        self._concat   = ConcatDataset(datasets)
        self.samples   = [s for ds in datasets for s in ds.samples]
        total = len(self.samples)
        n_g   = sum(1 for _, l in self.samples if l == 1)
        print(f"[dataset] MultiDirDataset ({split}): {total} total — glaucoma={n_g}, normal={total-n_g}")
        for ds in datasets:
            print(f"          {ds}")

    def __len__(self):       return len(self._concat)
    def __getitem__(self, i): return self._concat[i]

    @property
    def class_weights(self) -> torch.Tensor:
        counts = [sum(1 for _, l in self.samples if l == i) for i in range(2)]
        total  = sum(counts)
        return torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_train_dataset(data_root: str | Path, transform=None) -> MultiDirDataset:
    """
    Build the full training set from:
      - LAG train + LAG test
      - Fundus Train
      - ACRIMA
      - ORIGA
    """
    root = Path(data_root)
    tf   = transform or build_transforms("train")
    return MultiDirDataset([
        GlaucomaDataset(root / "train",   split="train", transform=tf),
        GlaucomaDataset(root / "test",    split="train", transform=tf),
        FolderGlaucomaDataset(root / "Fundus" / "Train",   split="train", transform=tf),
        GlaucomaDataset(root / "ACRIMA",  split="train", transform=tf),
        CSVGlaucomaDataset(root / "ORIGA", root / "origa_labels.csv", split="train", transform=tf),
    ], split="train")


def build_val_dataset(data_root: str | Path, transform=None) -> MultiDirDataset:
    """
    Build the validation set from:
      - LAG validation
      - Fundus Validation
    """
    root = Path(data_root)
    tf   = transform or build_transforms("validation")
    return MultiDirDataset([
        GlaucomaDataset(root / "validation",           split="validation", transform=tf),
        FolderGlaucomaDataset(root / "Fundus" / "Validation", split="validation", transform=tf),
    ], split="validation")
