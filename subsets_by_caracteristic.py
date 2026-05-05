from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

INPUT_CSV_CANDIDATES = [
    "datasets/jraigs_prepared/labels.csv",
    "datasets/jraigs_kept/labels.csv",
]
OUTPUT_ROOT = "datasets/jraigs_subsets"

CHARACTERISTICS = [
    "ANRS", "ANRI", "RNFLDS", "RNFLDI",
    "BCLVS", "BCLVI", "NVT", "DH", "LD", "LC",
]

NON_GLAUCOMA_LABELS = {
    "NRG",
    "0",
    "NEGATIVE",
    "NORMAL",
    "NON_GLAUCOMA",
    "NON-GLAUCOMA",
    "NO_GLAUCOMA",
}

GLAUCOMA_LABELS = {
    "RG",
    "1",
    "POSITIVE",
    "GLAUCOMA",
}

LABEL_COLUMN_CANDIDATES = [
    "Final Label",
    "label",
    "Label",
    "final_label",
]


def link_or_copy(src: str | None, dst: Path) -> bool:
    if not src:
        return False
    src_path = Path(src)
    if not src_path.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src_path, dst)
        except OSError:
            shutil.copy2(src_path, dst)
    return True


def write_subset_with_images(subset: pd.DataFrame, output_dir: Path) -> int:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    kept_images: list[str | None] = []
    for src in subset.get("kept_image", [None] * len(subset)):
        dst_name = Path(src).name if src else None
        dst = images_dir / dst_name if dst_name else None
        if dst and link_or_copy(src, dst):
            kept_images.append(str(dst))
        else:
            kept_images.append(None)

    subset = subset.copy()
    subset["subset_image"] = kept_images
    subset.to_csv(output_dir / "labels.csv", index=False)
    return len(subset)


def build_subset(df: pd.DataFrame, char: str, output_dir: Path) -> int:
    g1_col = f"G1 {char}"
    g2_col = f"G2 {char}"

    if g1_col not in df.columns or g2_col not in df.columns:
        print(f"  [SKIP] columns {g1_col} / {g2_col} not found")
        return 0

    subset = df[
        (df[g1_col] == 1)
        & (df[g2_col] == 1)
    ].copy()
    return write_subset_with_images(subset, output_dir)


def build_non_glaucoma_subset(df: pd.DataFrame, output_root: Path) -> int:
    label_col = next((col for col in LABEL_COLUMN_CANDIDATES if col in df.columns), None)
    if label_col is None:
        print("  [SKIP] label column not found for non-glaucoma subset")
        return 0

    label_values = df[label_col].astype(str).str.strip().str.upper()
    subset = df[label_values.isin(NON_GLAUCOMA_LABELS)].copy()

    out_dir = output_root / "non_glaucoma"
    n = write_subset_with_images(subset, out_dir)
    print(f"  non_glaucoma: {n} rows -> {out_dir}")
    return n


def build_glaucoma_subset(df: pd.DataFrame, output_root: Path) -> int:
    label_col = next((col for col in LABEL_COLUMN_CANDIDATES if col in df.columns), None)
    if label_col is None:
        print("  [SKIP] label column not found for glaucoma subset")
        return 0

    out_dir = output_root / "glaucoma"
    existing_csv = out_dir / "labels.csv"
    if existing_csv.exists():
        print(f"  glaucoma already exists, skipped: {existing_csv}")
        return 0

    label_values = df[label_col].astype(str).str.strip().str.upper()

    # Prefer explicit glaucoma labels; fallback to known non-glaucoma complement.
    glaucoma_mask = label_values.isin(GLAUCOMA_LABELS)
    if not glaucoma_mask.any():
        glaucoma_mask = label_values.ne("") & ~label_values.isin(NON_GLAUCOMA_LABELS)

    subset = df[glaucoma_mask].copy()

    n = write_subset_with_images(subset, out_dir)
    print(f"  glaucoma: {n} rows -> {out_dir}")
    return n


def build_brightness_subsets(df: pd.DataFrame, output_root: Path) -> None:
    if "brightness" not in df.columns:
        print("  [SKIP] brightness column not found")
        return

    bright_df = df[df["brightness"].notna()].copy()
    if bright_df.empty:
        print("  [SKIP] no non-empty brightness values")
        return

    q1 = bright_df["brightness"].quantile(0.33)
    q2 = bright_df["brightness"].quantile(0.66)

    low = bright_df[bright_df["brightness"] <= q1]
    medium = bright_df[(bright_df["brightness"] > q1) & (bright_df["brightness"] <= q2)]
    high = bright_df[bright_df["brightness"] > q2]

    for name, subset in [("low", low), ("medium", medium), ("high", high)]:
        out_dir = output_root / f"brightness_{name}"
        n = write_subset_with_images(subset, out_dir)
        print(f"  brightness_{name}: {n} rows -> {out_dir}")


def parse_color_triplet(value: object) -> tuple[float, float, float] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        return None
    try:
        r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None
    return r, g, b


def build_color_subsets(df: pd.DataFrame, output_root: Path) -> None:
    if "color_mean_rgb" not in df.columns:
        print("  [SKIP] color_mean_rgb column not found")
        return

    color_df = df.copy()
    rgb = color_df["color_mean_rgb"].apply(parse_color_triplet)
    color_df["_r"] = rgb.apply(lambda t: t[0] if t is not None else float("nan"))
    color_df["_g"] = rgb.apply(lambda t: t[1] if t is not None else float("nan"))
    color_df["_b"] = rgb.apply(lambda t: t[2] if t is not None else float("nan"))

    valid = color_df[color_df[["_r", "_g", "_b"]].notna().all(axis=1)]
    if valid.empty:
        print("  [SKIP] no valid color_mean_rgb values")
        return

    red = valid[(valid["_r"] > valid["_g"]) & (valid["_r"] > valid["_b"])]
    green = valid[(valid["_g"] > valid["_r"]) & (valid["_g"] > valid["_b"])]
    blue = valid[(valid["_b"] > valid["_r"]) & (valid["_b"] > valid["_g"])]

    for name, subset in [("red", red), ("green", green), ("blue", blue)]:
        out_dir = output_root / f"color_{name}"
        n = write_subset_with_images(subset.drop(columns=["_r", "_g", "_b"], errors="ignore"), out_dir)
        print(f"  color_{name}: {n} rows -> {out_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_csv = None
    for candidate in INPUT_CSV_CANDIDATES:
        candidate_path = project_root / candidate
        if candidate_path.exists():
            input_csv = candidate_path
            break

    if input_csv is None:
        joined = ", ".join(INPUT_CSV_CANDIDATES)
        raise FileNotFoundError(f"None of the input CSVs exist: {joined}")

    output_root = project_root / OUTPUT_ROOT

    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} rows from {input_csv}")

    for char in CHARACTERISTICS:
        output_dir = output_root / char
        n = build_subset(df, char, output_dir)
        print(f"  {char}: {n} rows -> {output_dir}")

    build_glaucoma_subset(df, output_root)
    build_non_glaucoma_subset(df, output_root)

    # build_brightness_subsets(df, output_root)
    build_color_subsets(df, output_root)

    print(f"\nAll subsets saved under {output_root}")


if __name__ == "__main__":
    main()
