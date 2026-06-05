from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = Path("outputs/jraigs_prepared.csv")
OUTPUT_DIR = Path("outputs")
BRIGHTNESS_HISTOGRAM = OUTPUT_DIR / "brightness_histogram.png"
COLOR_HISTOGRAM = OUTPUT_DIR / "color_histogram_rgb.png"


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


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def plot_brightness_histogram(df: pd.DataFrame, output_path: Path) -> None:
    if "brightness" not in df.columns:
        raise KeyError("Column 'brightness' not found in dataframe")

    brightness = pd.to_numeric(df["brightness"], errors="coerce").dropna()
    if brightness.empty:
        raise ValueError("No valid brightness values found")

    plt.figure(figsize=(10, 6))
    plt.hist(brightness, bins=40, color="#577590", edgecolor="black", alpha=0.85)
    plt.xlabel("Brightness")
    plt.ylabel("Frequency")
    plt.title("Brightness Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_color_histogram(df: pd.DataFrame, output_path: Path) -> None:
    if "color_mean_rgb" not in df.columns:
        raise KeyError("Column 'color_mean_rgb' not found in dataframe")

    rgb_triplets = df["color_mean_rgb"].apply(parse_color_triplet)
    valid_rgb = rgb_triplets[rgb_triplets.notna()]
    if valid_rgb.empty:
        raise ValueError("No valid color_mean_rgb values found")

    rgb_df = pd.DataFrame(valid_rgb.tolist(), columns=["red", "green", "blue"])

    plt.figure(figsize=(10, 6))
    plt.hist(rgb_df["red"], bins=40, color="#D62828", alpha=0.45, label="Red")
    plt.hist(rgb_df["green"], bins=40, color="#2A9D8F", alpha=0.45, label="Green")
    plt.hist(rgb_df["blue"], bins=40, color="#277DA1", alpha=0.45, label="Blue")
    plt.xlabel("Mean Channel Value")
    plt.ylabel("Frequency")
    plt.title("RGB Mean Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe(INPUT_CSV)

    plot_brightness_histogram(df, BRIGHTNESS_HISTOGRAM)
    plot_color_histogram(df, COLOR_HISTOGRAM)

    print(f"Saved plot: {BRIGHTNESS_HISTOGRAM}")
    print(f"Saved plot: {COLOR_HISTOGRAM}")


if __name__ == "__main__":
    main()

