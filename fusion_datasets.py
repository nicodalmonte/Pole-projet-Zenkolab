from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    if text in {"RG", "NRG", "UR"}:
        return text
    return text


def load_jraigs_labels(csv_path: Path) -> pd.DataFrame:
    # JRAIGS labels are semicolon-separated.
    df = pd.read_csv(csv_path, sep=";")

    required_columns = {"Eye ID", "Final Label", "Label G1", "Label G2"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required JRAIGS columns: {missing_str}")

    out = df.copy()
    out["filename"] = out["Eye ID"].astype(str)
    out["label"] = out["Final Label"].map(normalize_label)
    out["label_g1"] = out["Label G1"].map(normalize_label)
    out["label_g2"] = out["Label G2"].map(normalize_label)

    return out


def build_image_index(images_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem.upper()
        if stem not in image_index:
            image_index[stem] = path
    return image_index


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_kept_images(
    filtered_df: pd.DataFrame,
    images_root: Path,
    kept_root: Path,
) -> tuple[pd.DataFrame, int]:
    image_index = build_image_index(images_root)
    kept_images_dir = kept_root / "images"
    kept_rows: list[dict[str, object]] = []
    missing_count = 0

    for _, row in filtered_df.iterrows():
        eye_id = str(row["filename"]).strip().upper()
        src = image_index.get(eye_id)
        if src is None:
            missing_count += 1
            continue

        dst = kept_images_dir / src.name
        if not dst.exists():
            safe_link_or_copy(src, dst)

        row_dict = row.to_dict()
        row_dict["source_image"] = str(src)
        row_dict["kept_image"] = str(dst)
        kept_rows.append(row_dict)

    return pd.DataFrame(kept_rows), missing_count


def filter_jraigs(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        (df["label"].notna())
        & (df["label"] != "UR")
        & (df["label_g1"].notna())
        & (df["label_g2"].notna())
        & (df["label_g1"] == df["label_g2"])
    ].copy()
    return filtered


def drop_requested_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Explicitly remove grader label columns requested by user.
    cleaned = cleaned.drop(columns=["Label G1", "Label G2", "Label G3"], errors="ignore")

    # Remove rows where all G3 characteristic columns are non-empty (3rd grader rows).
    g3_cols = [c for c in cleaned.columns if c.startswith("G3 ")]
    if g3_cols:
        has_all_g3 = (
            cleaned[g3_cols].notna().all(axis=1)
            & cleaned[g3_cols].astype(str).apply(lambda s: s.str.strip().ne("")).all(axis=1)
        )
        cleaned = cleaned[~has_all_g3]

    # Drop all G3 characteristic columns.
    cleaned = cleaned.drop(columns=g3_cols, errors="ignore")

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JRAIGS labels with label != UR and labelG1 == labelG2"
    )
    parser.add_argument(
        "--input",
        default="datasets/manit2022__jraigs-dataset/JustRAIGS_Train_labels.csv",
        help="Path to JRAIGS labels CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/jraigs_labels_g1_eq_g2_no_ur.csv",
        help="Path to output filtered CSV",
    )
    parser.add_argument(
        "--images-root",
        default="datasets/manit2022__jraigs-dataset",
        help="Root folder where JRAIGS image files are stored",
    )
    parser.add_argument(
        "--kept-root",
        default="datasets/jraigs_kept",
        help="Folder to create with kept images and corresponding CSV",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_csv = project_root / args.input
    output_csv = project_root / args.output
    images_root = project_root / args.images_root
    kept_root = project_root / args.kept_root

    if not input_csv.exists():
        raise FileNotFoundError(f"JRAIGS labels file not found: {input_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"JRAIGS images root not found: {images_root}")

    raw_df = load_jraigs_labels(input_csv)
    filtered_df = filter_jraigs(raw_df)
    filtered_df = drop_requested_columns(filtered_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)

    kept_root.mkdir(parents=True, exist_ok=True)
    kept_df, missing_count = export_kept_images(filtered_df, images_root, kept_root)
    kept_csv = kept_root / "labels.csv"
    kept_df.to_csv(kept_csv, index=False)

    print(f"Input rows scanned: {len(raw_df)}")
    print(f"Rows kept after filtering: {len(filtered_df)}")
    print(f"Saved filtered JRAIGS CSV to: {output_csv}")
    print(f"Kept images exported: {len(kept_df)}")
    print(f"Missing images skipped: {missing_count}")
    print(f"Kept dataset folder: {kept_root}")
    print(f"Kept dataset CSV: {kept_csv}")


if __name__ == "__main__":
    main()
