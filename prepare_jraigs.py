from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    import torch
except ImportError:
    torch = None

INPUT_CSV = "datasets/manit2022__jraigs-dataset/JustRAIGS_Train_labels.csv"
OUTPUT_CSV = "outputs/jraigs_prepared.csv"
IMAGES_ROOT = "datasets/manit2022__jraigs-dataset"
KEPT_IMAGES_DIR = "datasets/jraigs_prepared/images"

COLUMNS_TO_DROP = ["Label G1", "Label G2", "Label G3"]


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text if text else None


def prepare(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    # --- Load ---
    df = pd.read_csv(input_csv, sep=";", low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- Normalize grader labels ---
    df["_label"] = df["Final Label"].map(normalize_label)
    df["_label_g1"] = df["Label G1"].map(normalize_label)
    df["_label_g2"] = df["Label G2"].map(normalize_label)

    # --- Filter: keep label != UR and label_g1 == label_g2 ---
    df = df[
        df["_label"].notna()
        & (df["_label"] != "UR")
        & df["_label_g1"].notna()
        & df["_label_g2"].notna()
        & (df["_label_g1"] == df["_label_g2"])
    ].copy()
    print(f"After filter (not UR, G1==G2): {len(df)} rows")

    # --- Drop helper columns ---
    df = df.drop(columns=["_label", "_label_g1", "_label_g2"])

    # --- Drop Label G1, Label G2, Label G3 ---
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # --- Drop rows where all G3 characteristic columns are non-empty ---
    g3_cols = [c for c in df.columns if c.startswith("G3 ")]
    if g3_cols:
        has_all_g3 = (
            df[g3_cols].notna().all(axis=1)
            & df[g3_cols].astype(str).apply(lambda s: s.str.strip().ne("")).all(axis=1)
        )
        removed = has_all_g3.sum()
        df = df[~has_all_g3]
        print(f"Removed {removed} rows with non-empty G3 columns")

    # --- Drop all G3 characteristic columns ---
    df = df.drop(columns=g3_cols, errors="ignore")
    print(f"Dropped {len(COLUMNS_TO_DROP) + len(g3_cols)} columns, {len(df.columns)} remaining")

    # --- Save ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    return df


def build_image_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file():
            stem = path.stem.upper()
            if stem not in index:
                index[stem] = path
    return index


def export_images(df: pd.DataFrame, images_root: Path, kept_images_dir: Path) -> pd.DataFrame:
    kept_images_dir.mkdir(parents=True, exist_ok=True)
    index = build_image_index(images_root)
    source_images, kept_images = [], []
    missing = 0

    for eye_id in df["Eye ID"].astype(str):
        src = index.get(eye_id.strip().upper())
        if src is None:
            missing += 1
            source_images.append(None)
            kept_images.append(None)
            continue
        dst = kept_images_dir / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        source_images.append(str(src))
        kept_images.append(str(dst))

    df = df.copy()
    df["source_image"] = source_images
    df["kept_image"] = kept_images
    print(f"Images exported: {len(df) - missing}  |  missing: {missing}")
    return df


def add_image_stats(df: pd.DataFrame, image_col: str = "kept_image") -> pd.DataFrame:
    brightness_values: list[float] = []
    color_mean_values: list[str] = []

    use_gpu = torch is not None and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Image stats device: {device}")

    for image_path in df[image_col].tolist():
        if not image_path:
            brightness_values.append(float("nan"))
            color_mean_values.append("")
            continue

        try:
            with Image.open(image_path) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
                if use_gpu:
                    rgb_tensor = torch.from_numpy(rgb).to("cuda", non_blocking=True)
                    brightness = float(rgb_tensor.mean().item())
                    rgb_mean_tensor = rgb_tensor.view(-1, 3).mean(dim=0)
                    rgb_mean = rgb_mean_tensor.detach().cpu().numpy()
                else:
                    brightness = float(rgb.mean())
                    rgb_mean = rgb.reshape(-1, 3).mean(axis=0)
                color_mean = f"{rgb_mean[0]:.2f},{rgb_mean[1]:.2f},{rgb_mean[2]:.2f}"
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
            brightness = float("nan")
            color_mean = ""

        brightness_values.append(brightness)
        color_mean_values.append(color_mean)

    out = df.copy()
    out["brightness"] = brightness_values
    out["color_mean_rgb"] = color_mean_values
    return out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    output_csv_path = project_root / OUTPUT_CSV
    df = prepare(project_root / INPUT_CSV, output_csv_path)
    df = export_images(
        df,
        project_root / IMAGES_ROOT,
        project_root / KEPT_IMAGES_DIR,
    )
    df = add_image_stats(df, image_col="kept_image")

    # Overwrite output CSV so it also contains image paths + image stats.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Updated output CSV with image stats: {output_csv_path}")

    # Save final CSV with image paths alongside the images folder
    final_csv = project_root / "datasets" / "jraigs_prepared" / "labels.csv"
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_csv, index=False)
    print(f"Final CSV with image paths: {final_csv}")
