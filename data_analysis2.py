from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
JRAIGS_DEFAULT_PATH = Path("datasets") / "manit2022__jraigs-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all images in a folder and generate plots")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("datasets"),
        help="Folder to scan recursively for images",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs") / "folder_eda",
        help="Directory where CSV and plots are saved",
    )
    parser.add_argument(
        "--jraigs",
        action="store_true",
        help="Use datasets/manit2022__jraigs-dataset as input folder",
    )
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(images)


def analyze_image(path: Path) -> dict[str, float | int | str] | None:
    try:
        with Image.open(path) as img:
            arr = np.array(img)
            width, height = img.size
            mode = img.mode

        if arr.ndim == 2:
            channels = 1
            gray = arr.astype(np.float32)
            red_mean = np.nan
            green_mean = np.nan
            blue_mean = np.nan
        else:
            channels = arr.shape[2]
            rgb = arr[:, :, :3].astype(np.float32)
            gray = rgb.mean(axis=2)
            red_mean = float(rgb[:, :, 0].mean())
            green_mean = float(rgb[:, :, 1].mean())
            blue_mean = float(rgb[:, :, 2].mean())

        return {
            "path": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "mode": mode,
            "width": int(width),
            "height": int(height),
            "pixel_count": int(width * height),
            "channels": int(channels),
            "brightness_mean": float(gray.mean()),
            "contrast_std": float(gray.std()),
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
        }
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def save_plots(df: pd.DataFrame, outdir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["width"], df["height"], alpha=0.45, s=18)
    ax.set_title("Image Resolution Distribution")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.tight_layout()
    fig.savefig(outdir / "resolution_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["brightness_mean"], bins=40, alpha=0.8)
    ax.set_title("Brightness Histogram")
    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outdir / "brightness_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ext_counts = df["extension"].value_counts().sort_values(ascending=False)
    ax.bar(ext_counts.index, ext_counts.values)
    ax.set_title("File Extension Counts")
    ax.set_xlabel("Extension")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outdir / "extension_counts.png", dpi=150)
    plt.close(fig)

    rgb_df = df.dropna(subset=["red_mean", "green_mean", "blue_mean"])
    if not rgb_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (col, color, title) in enumerate(
            [
                ("red_mean", "red", "Red Channel Mean"),
                ("green_mean", "green", "Green Channel Mean"),
                ("blue_mean", "blue", "Blue Channel Mean"),
            ]
        ):
            axes[idx].hist(rgb_df[col], bins=40, color=color, alpha=0.7)
            axes[idx].set_title(title)
            axes[idx].set_xlabel("Pixel Value")
            axes[idx].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(outdir / "rgb_channel_histograms.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    folder = (JRAIGS_DEFAULT_PATH if args.jraigs else args.folder).resolve()
    outdir = args.outdir.resolve()

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(folder)
    if not image_paths:
        raise RuntimeError(f"No images found in folder: {folder}")

    rows = []
    for path in image_paths:
        stats = analyze_image(path)
        if stats is not None:
            rows.append(stats)

    if not rows:
        raise RuntimeError("No readable images were found.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "all_images_analysis.csv", index=False)
    save_plots(df, outdir)

    print("Analysis completed")
    print(f"Folder scanned: {folder}")
    print(f"Images analyzed: {len(df)}")
    print(f"CSV: {outdir / 'all_images_analysis.csv'}")
    print(f"Plots saved in: {outdir}")

    summary = df[["width", "height", "brightness_mean", "contrast_std"]].describe().round(2)
    print("\nSummary statistics:")
    print(summary)


if __name__ == "__main__":
    main()
