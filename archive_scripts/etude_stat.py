from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()
INPUT_CSV = PROJECT_ROOT / "datasets" / "jraigs_prepared" / "labels.csv"
OUTPUT_DIR=PROJECT_ROOT/"outputs"

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
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def dominant_color(rgb_triplet: tuple[float, float, float] | None) -> str | None:
    if rgb_triplet is None:
        return None
    r, g, b = rgb_triplet
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


def describe_series(series: pd.Series, label: str) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame([
            {
                "subset": label,
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "q25": float("nan"),
                "median": float("nan"),
                "q75": float("nan"),
                "max": float("nan"),
            }
        ])
    return pd.DataFrame([
        {
            "subset": label,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }
    ])


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded: {INPUT_CSV}")
    print(f"Rows={len(df)} | Cols={len(df.columns)}")

    if "Final Label" not in df.columns:
        raise KeyError("Column 'Final Label' is missing from the CSV")

    labels = df["Final Label"].astype(str).str.strip().str.upper()
    glaucoma_mask = labels.isin(GLAUCOMA_LABELS)
    non_glaucoma_mask = labels.isin(NON_GLAUCOMA_LABELS)

    df_glaucoma = df[glaucoma_mask].copy()
    df_non_glaucoma = df[non_glaucoma_mask].copy()

    print(f"Glaucoma rows: {len(df_glaucoma)}")
    print(f"Non-glaucoma rows: {len(df_non_glaucoma)}")

    b_glaucoma = pd.to_numeric(df_glaucoma.get("brightness"), errors="coerce").dropna()
    b_non_glaucoma = pd.to_numeric(df_non_glaucoma.get("brightness"), errors="coerce").dropna()

    blur_glaucoma = pd.to_numeric(df_glaucoma.get("blur_score"), errors="coerce").dropna()
    blur_non_glaucoma = pd.to_numeric(df_non_glaucoma.get("blur_score"), errors="coerce").dropna()

    rgb_glaucoma = df_glaucoma.get("color_mean_rgb", pd.Series([None] * len(df_glaucoma))).apply(parse_color_triplet)
    rgb_non_glaucoma = df_non_glaucoma.get("color_mean_rgb", pd.Series([None] * len(df_non_glaucoma))).apply(parse_color_triplet)

    dominant_glaucoma = rgb_glaucoma.apply(dominant_color).dropna()
    dominant_non_glaucoma = rgb_non_glaucoma.apply(dominant_color).dropna()

    colors = ["red", "green", "blue"]
    color_count_glaucoma = dominant_glaucoma.value_counts().reindex(colors, fill_value=0)
    color_count_non_glaucoma = dominant_non_glaucoma.value_counts().reindex(colors, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].hist(b_glaucoma, bins=40, alpha=0.6, label="glaucoma", color="#d62828", edgecolor="black",density=True)
    axes[0].hist(b_non_glaucoma, bins=40, alpha=0.6, label="non_glaucoma", color="#2a9d8f", edgecolor="black",density=True)
    axes[0].set_title("Brightness distribution")
    axes[0].set_xlabel("brightness")
    axes[0].set_ylabel("frequency")
    axes[0].legend()

    axes[1].hist(blur_glaucoma, bins=40, alpha=0.6, label="glaucoma", color="#f77f00", edgecolor="black",density=True)
    axes[1].hist(blur_non_glaucoma, bins=40, alpha=0.6, label="non_glaucoma", color="#003049", edgecolor="black",density=True)
    axes[1].set_title("Blur score distribution")
    axes[1].set_xlabel("blur_score")
    axes[1].set_ylabel("frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_g_ng.png", dpi=150)
    plt.close()

        # Extraire les valeurs R, G, B séparément
    def extract_channel(rgb_series, channel):
        idx = {"r": 0, "g": 1, "b": 2}[channel]
        return rgb_series.dropna().apply(lambda t: t[idx])
    
    # --- Deuxième figure : 3 graphes côte à côte ---
    r_g = extract_channel(rgb_glaucoma, "r")
    g_g = extract_channel(rgb_glaucoma, "g")
    b_g = extract_channel(rgb_glaucoma, "b")

    r_ng = extract_channel(rgb_non_glaucoma, "r")
    g_ng = extract_channel(rgb_non_glaucoma, "g")
    b_ng = extract_channel(rgb_non_glaucoma, "b")

    # --- Deuxième figure : distributions R, G, B ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 5))

    channels = [
        ("Red channel", r_g, r_ng, "#d62828", "#f4a0a0"),
        ("Green channel", g_g, g_ng, "#2a9d8f", "#a8e6df"),
        ("Blue channel", b_g, b_ng, "#264653", "#90b4c1"),
    ]

    for ax, (title, glau, non_glau, c_g, c_ng) in zip(axes2, channels):
        w_g = np.ones(len(glau)) / len(glau) * 100
        w_ng = np.ones(len(non_glau)) / len(non_glau) * 100
        ax.hist(glau, bins=40, alpha=0.6, label="glaucoma", color=c_g, edgecolor="black", weights=w_g)
        ax.hist(non_glau, bins=40, alpha=0.6, label="non_glaucoma", color=c_ng, edgecolor="black", weights=w_ng)
        ax.set_title(title)
        ax.set_xlabel("mean value")
        ax.set_ylabel("percentage (%)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rgb_distributions_g_ng.png", dpi=150)
    plt.close()

    brightness_stats = pd.concat(
        [
            describe_series(b_glaucoma, "glaucoma"),
            describe_series(b_non_glaucoma, "non_glaucoma"),
        ],
        ignore_index=True,
    )

    blur_stats = pd.concat(
        [
            describe_series(blur_glaucoma, "glaucoma"),
            describe_series(blur_non_glaucoma, "non_glaucoma"),
        ],
        ignore_index=True,
    )

    color_stats = pd.DataFrame(
        {
            "subset": ["glaucoma", "non_glaucoma"],
            "valid_color_count": [int(len(dominant_glaucoma)), int(len(dominant_non_glaucoma))],
            "red_pct": [
                float((color_count_glaucoma["red"] / len(dominant_glaucoma) * 100) if len(dominant_glaucoma) else 0.0),
                float((color_count_non_glaucoma["red"] / len(dominant_non_glaucoma) * 100) if len(dominant_non_glaucoma) else 0.0),
            ],
            "green_pct": [
                float((color_count_glaucoma["green"] / len(dominant_glaucoma) * 100) if len(dominant_glaucoma) else 0.0),
                float((color_count_non_glaucoma["green"] / len(dominant_non_glaucoma) * 100) if len(dominant_non_glaucoma) else 0.0),
            ],
            "blue_pct": [
                float((color_count_glaucoma["blue"] / len(dominant_glaucoma) * 100) if len(dominant_glaucoma) else 0.0),
                float((color_count_non_glaucoma["blue"] / len(dominant_non_glaucoma) * 100) if len(dominant_non_glaucoma) else 0.0),
            ],
        }
    )

    print("\nBrightness statistics")
    print(brightness_stats.to_string(index=False))

    print("\nBlur score statistics")
    print(blur_stats.to_string(index=False))

    print("\nDominant color statistics (%)")
    print(color_stats.to_string(index=False))


if __name__ == "__main__":
    main()
