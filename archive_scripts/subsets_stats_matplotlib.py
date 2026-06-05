from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUBSETS_ROOT = Path("datasets/jraigs_subsets")
OUTPUT_DIR = Path("outputs")
SUMMARY_CSV = OUTPUT_DIR / "subset_stats_summary.csv"
ROWS_PLOT = OUTPUT_DIR / "subset_rows_by_group.png"
LABEL_PLOT = OUTPUT_DIR / "subset_label_distribution.png"
MISSING_PLOT = OUTPUT_DIR / "subset_missing_images_rate.png"
BRIGHTNESS_MEAN_PLOT = OUTPUT_DIR / "subset_brightness_mean.png"
BRIGHTNESS_VARIANCE_PLOT = OUTPUT_DIR / "subset_brightness_variance.png"
COLOR_MEAN_PLOT = OUTPUT_DIR / "subset_color_mean_rgb.png"
COLOR_VARIANCE_PLOT = OUTPUT_DIR / "subset_color_variance_rgb.png"


def resolve_image_col(df: pd.DataFrame) -> str | None:
    for col in ("subset_image", "kept_image", "source_image"):
        if col in df.columns:
            return col
    return None


def resolve_label_col(df: pd.DataFrame) -> str | None:
    for col in ("Final Label", "label", "Label", "final_label"):
        if col in df.columns:
            return col
    return None


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
        red, green, blue = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None

    return red, green, blue


def image_exists(path_str: object) -> bool:
    if pd.isna(path_str):
        return False
    path = str(path_str).strip()
    if not path:
        return False
    return Path(path).exists()


def collect_subset_stats(subset_dir: Path) -> dict[str, object] | None:
    labels_path = subset_dir / "labels.csv"
    if not labels_path.exists():
        return None

    df = pd.read_csv(labels_path, low_memory=False)
    n_rows = len(df)

    image_col = resolve_image_col(df)
    images_present = 0
    missing_images = 0
    if image_col is not None and n_rows > 0:
        exists_flags = df[image_col].apply(image_exists)
        images_present = int(exists_flags.sum())
        missing_images = int((~exists_flags).sum())

    label_col = resolve_label_col(df)
    nrg_count = 0
    rg_count = 0
    other_count = 0
    if label_col is not None and n_rows > 0:
        labels = df[label_col].astype(str).str.strip().str.upper()
        nrg_count = int((labels == "NRG").sum())
        rg_count = int((labels == "RG").sum())
        other_count = int(n_rows - nrg_count - rg_count)

    brightness_mean = float("nan")
    brightness_median = float("nan")
    brightness_variance = float("nan")
    if "brightness" in df.columns:
        brightness = pd.to_numeric(df["brightness"], errors="coerce")
        if brightness.notna().any():
            brightness_mean = float(brightness.mean())
            brightness_median = float(brightness.median())
            brightness_variance = float(brightness.var())

    red_mean = float("nan")
    green_mean = float("nan")
    blue_mean = float("nan")
    red_variance = float("nan")
    green_variance = float("nan")
    blue_variance = float("nan")
    if "color_mean_rgb" in df.columns:
        rgb_triplets = df["color_mean_rgb"].apply(parse_color_triplet)
        valid_rgb = rgb_triplets[rgb_triplets.notna()]
        if not valid_rgb.empty:
            rgb_df = pd.DataFrame(valid_rgb.tolist(), columns=["red", "green", "blue"])
            red_mean = float(rgb_df["red"].mean())
            green_mean = float(rgb_df["green"].mean())
            blue_mean = float(rgb_df["blue"].mean())
            red_variance = float(rgb_df["red"].var())
            green_variance = float(rgb_df["green"].var())
            blue_variance = float(rgb_df["blue"].var())

    return {
        "subset": subset_dir.name,
        "rows": n_rows,
        "images_present": images_present,
        "missing_images": missing_images,
        "missing_images_rate": (missing_images / n_rows) if n_rows > 0 else 0.0,
        "nrg": nrg_count,
        "rg": rg_count,
        "other_labels": other_count,
        "brightness_mean": brightness_mean,
        "brightness_median": brightness_median,
        "brightness_variance": brightness_variance,
        "red_mean": red_mean,
        "green_mean": green_mean,
        "blue_mean": blue_mean,
        "red_variance": red_variance,
        "green_variance": green_variance,
        "blue_variance": blue_variance,
    }


def save_plots(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    sorted_df = summary_df.sort_values("rows", ascending=False)
    x_positions = range(len(sorted_df))
    bar_width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_df["subset"], sorted_df["rows"], color="#2A9D8F")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rows")
    plt.title("Rows Per Subset")
    plt.tight_layout()
    plt.savefig(ROWS_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_df["subset"], sorted_df["nrg"], label="NRG", color="#264653")
    plt.bar(sorted_df["subset"], sorted_df["rg"], bottom=sorted_df["nrg"], label="RG", color="#E76F51")
    plt.bar(
        sorted_df["subset"],
        sorted_df["other_labels"],
        bottom=sorted_df["nrg"] + sorted_df["rg"],
        label="Other",
        color="#E9C46A",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rows")
    plt.title("Label Distribution Per Subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LABEL_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_df["subset"], sorted_df["missing_images_rate"] * 100.0, color="#F4A261")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing Image Rate (%)")
    plt.title("Missing Images Rate Per Subset")
    plt.tight_layout()
    plt.savefig(MISSING_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_df["subset"], sorted_df["brightness_mean"], color="#577590")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Brightness Mean")
    plt.title("Brightness Mean Per Subset")
    plt.tight_layout()
    plt.savefig(BRIGHTNESS_MEAN_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_df["subset"], sorted_df["brightness_variance"], color="#43AA8B")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Brightness Variance")
    plt.title("Brightness Variance Per Subset")
    plt.tight_layout()
    plt.savefig(BRIGHTNESS_VARIANCE_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar([x - bar_width for x in x_positions], sorted_df["red_mean"], width=bar_width, label="Red", color="#D62828")
    plt.bar(x_positions, sorted_df["green_mean"], width=bar_width, label="Green", color="#2A9D8F")
    plt.bar([x + bar_width for x in x_positions], sorted_df["blue_mean"], width=bar_width, label="Blue", color="#277DA1")
    plt.xticks(list(x_positions), sorted_df["subset"], rotation=45, ha="right")
    plt.ylabel("Channel Mean")
    plt.title("RGB Mean Per Subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLOR_MEAN_PLOT, dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.bar([x - bar_width for x in x_positions], sorted_df["red_variance"], width=bar_width, label="Red", color="#D62828")
    plt.bar(x_positions, sorted_df["green_variance"], width=bar_width, label="Green", color="#2A9D8F")
    plt.bar([x + bar_width for x in x_positions], sorted_df["blue_variance"], width=bar_width, label="Blue", color="#277DA1")
    plt.xticks(list(x_positions), sorted_df["subset"], rotation=45, ha="right")
    plt.ylabel("Channel Variance")
    plt.title("RGB Variance Per Subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLOR_VARIANCE_PLOT, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUBSETS_ROOT.exists():
        raise FileNotFoundError(f"Subsets root not found: {SUBSETS_ROOT}")

    stats_rows: list[dict[str, object]] = []
    for child in sorted(SUBSETS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        row = collect_subset_stats(child)
        if row is not None:
            stats_rows.append(row)

    summary_df = pd.DataFrame(stats_rows)
    summary_df = summary_df.sort_values("subset").reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    save_plots(summary_df)

    print(f"Saved summary: {SUMMARY_CSV}")
    print(f"Saved plot: {ROWS_PLOT}")
    print(f"Saved plot: {LABEL_PLOT}")
    print(f"Saved plot: {MISSING_PLOT}")
    print(f"Saved plot: {BRIGHTNESS_MEAN_PLOT}")
    print(f"Saved plot: {BRIGHTNESS_VARIANCE_PLOT}")
    print(f"Saved plot: {COLOR_MEAN_PLOT}")
    print(f"Saved plot: {COLOR_VARIANCE_PLOT}")
    print("\nTop subsets by rows:")
    print(summary_df.sort_values("rows", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
