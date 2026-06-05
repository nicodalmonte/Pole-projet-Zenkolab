from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

INPUT_CSV_CANDIDATES = [
    "datasets/jraigs_prepared/labels.csv",
    "datasets/jraigs_kept/labels.csv",
]
SUBSET_CONFIGS = [
    ("brightness_over_90", lambda brightness: brightness > 90.0, "brightness > 90"),
    ("brightness_under_20", lambda brightness: brightness < 25.0, "brightness < 20"),
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

    subset_images: list[str | None] = []
    for src in subset.get("kept_image", [None] * len(subset)):
        dst_name = Path(src).name if src else None
        dst = images_dir / dst_name if dst_name else None
        if dst and link_or_copy(src, dst):
            subset_images.append(str(dst))
        else:
            subset_images.append(None)

    subset = subset.copy()
    subset["subset_image"] = subset_images
    subset.to_csv(output_dir / "labels.csv", index=False)
    return len(subset)


def resolve_input_csv(project_root: Path) -> Path:
    for candidate in INPUT_CSV_CANDIDATES:
        candidate_path = project_root / candidate
        if candidate_path.exists():
            return candidate_path

    joined = ", ".join(INPUT_CSV_CANDIDATES)
    raise FileNotFoundError(f"None of the input CSVs exist: {joined}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_csv = resolve_input_csv(project_root)

    df = pd.read_csv(input_csv, low_memory=False)
    if "brightness" not in df.columns:
        raise KeyError("Column 'brightness' not found in input CSV")

    brightness = pd.to_numeric(df["brightness"], errors="coerce")
    print(f"Loaded {len(df)} rows from {input_csv}")

    for subset_name, condition, description in SUBSET_CONFIGS:
        output_dir = project_root / "datasets" / "jraigs_subsets" / subset_name
        subset = df[condition(brightness)].copy()
        subset["brightness"] = brightness.loc[subset.index]
        count = write_subset_with_images(subset, output_dir)
        print(f"{description}: {count} rows -> {output_dir}")


if __name__ == "__main__":
    main()