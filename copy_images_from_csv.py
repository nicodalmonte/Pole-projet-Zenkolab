from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


COMMON_IMAGE_COLUMNS = [
    "kept_image",
    "source_image",
    "image",
    "image_path",
    "filename",
    "file_name",
]


def detect_image_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            available = ", ".join(df.columns)
            raise KeyError(f"Column '{requested}' not found. Available columns: {available}")
        return requested

    for col in COMMON_IMAGE_COLUMNS:
        if col in df.columns:
            return col

    available = ", ".join(df.columns)
    raise KeyError(
        "No image column detected automatically. "
        f"Use --image-column. Available columns: {available}"
    )


def resolve_source_path(value: object, source_dir: Path) -> Path | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    given = Path(text)
    if given.is_absolute() and given.exists():
        return given

    # Try relative path first, then fallback to basename.
    candidate_rel = source_dir / text
    if candidate_rel.exists():
        return candidate_rel

    candidate_name = source_dir / given.name
    if candidate_name.exists():
        return candidate_name

    return None


def copy_images_from_csv(
    csv_path: Path,
    source_dir: Path,
    output_dir: Path,
    image_column: str | None,
) -> tuple[int, int, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    col = detect_image_column(df, image_column)

    rows = len(df)
    copied = 0
    missing = 0

    for value in df[col]:
        src = resolve_source_path(value, source_dir)
        if src is None:
            missing += 1
            continue

        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    return rows, copied, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images listed in a CSV into another dataset folder."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to CSV containing image paths or image file names.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Root folder where images currently exist.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination folder where selected images will be copied.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help="CSV column with image paths or names. Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_rows, copied, missing = copy_images_from_csv(
        csv_path=args.csv,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        image_column=args.image_column,
    )

    print(f"Rows in CSV: {total_rows}")
    print(f"Images copied: {copied}")
    print(f"Missing/unresolved image entries: {missing}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()