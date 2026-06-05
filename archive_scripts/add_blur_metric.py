from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def variance_of_laplacian(image_path: Path, thumb_size: int) -> float | None:
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((thumb_size, thumb_size), Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.float32)

        # 4-neighbor discrete Laplacian
        lap = (
            np.roll(arr, 1, axis=0)
            + np.roll(arr, -1, axis=0)
            + np.roll(arr, 1, axis=1)
            + np.roll(arr, -1, axis=1)
            - 4.0 * arr
        )
        return float(lap.var())
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return None


def resolve_image_col(df: pd.DataFrame, explicit_col: str | None) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise KeyError(f"Image column not found: {explicit_col}")
        return explicit_col

    for candidate in ("kept_image", "source_image"):
        if candidate in df.columns:
            return candidate

    raise KeyError("No image column found. Expected one of: kept_image, source_image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute blur metric (Variance of Laplacian) and save it in labels.csv",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("datasets/jraigs_prepared/labels.csv"),
        help="Input labels CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default=None,
        help="Image path column name (default: auto detect kept_image/source_image)",
    )
    parser.add_argument(
        "--blur-col",
        type=str,
        default="blur_score",
        help="Name of the blur score column to create/update",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=64,
        help="Resize image to N x N before metric computation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick tests",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute rows even when blur column already has a value",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parent
    input_csv = args.input_csv if args.input_csv.is_absolute() else project_root / args.input_csv
    output_csv = args.output_csv if args.output_csv else input_csv
    if not output_csv.is_absolute():
        output_csv = project_root / output_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    image_col = resolve_image_col(df, args.image_col)

    print(f"Loaded: {input_csv}")
    print(f"Rows={len(df)} | Cols={len(df.columns)} | image_col={image_col}")

    if args.blur_col not in df.columns:
        df[args.blur_col] = np.nan

    image_paths = df[image_col].astype(str)
    valid_mask = (
        df[image_col].notna()
        & image_paths.str.strip().ne("")
        & image_paths.apply(lambda p: Path(p).exists())
    )

    if not args.overwrite_existing:
        existing_mask = pd.to_numeric(df[args.blur_col], errors="coerce").notna()
        todo_mask = valid_mask & ~existing_mask
    else:
        todo_mask = valid_mask

    todo_idx = df.index[todo_mask]
    if args.limit is not None:
        todo_idx = todo_idx[: args.limit]

    print(f"Rows with valid image paths: {int(valid_mask.sum())}")
    print(f"Rows to compute blur metric: {len(todo_idx)}")

    for idx in tqdm(todo_idx, desc="blur"):
        score = variance_of_laplacian(Path(str(df.at[idx, image_col])), thumb_size=args.thumb_size)
        df.at[idx, args.blur_col] = score if score is not None else np.nan

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    computed = pd.to_numeric(df[args.blur_col], errors="coerce")
    print("\nSaved blur metric.")
    print(f"Output: {output_csv}")
    print(f"Column: {args.blur_col}")
    print(f"Non-null blur scores: {int(computed.notna().sum())}")
    if computed.notna().any():
        print(computed.describe())


if __name__ == "__main__":
    main()
