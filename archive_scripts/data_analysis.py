"""Command-line EDA for glaucoma datasets.

This script analyzes label coverage, class balance, image quality proxies,
and duplicate images across the configured datasets.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

try:
	import cv2  # type: ignore
except Exception:
	cv2 = None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run EDA over glaucoma datasets")
	parser.add_argument(
		"--root",
		type=Path,
		default=Path("/raid/home/students/hawky_luc/P-le-Projet-Zenkolab"),
		help="Project root path",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=None,
		help="Output directory for reports and figures (default: <root>/outputs/eda)",
	)
	parser.add_argument(
		"--max_images",
		type=int,
		default=None,
		help="Optional cap on number of rows processed for quick tests",
	)
	parser.add_argument(
		"--skip_plots",
		action="store_true",
		help="Skip generating and saving plot images",
	)
	parser.add_argument(
		"--sample_hashes",
		type=int,
		default=None,
		help="Optional cap for duplicate-hash step to speed up runs",
	)
	parser.add_argument(
		"--near_dup_distance",
		type=int,
		default=6,
		help="Hamming distance threshold for perceptual near-duplicates",
	)
	return parser.parse_args()


def build_csv_config(root: Path) -> list[dict[str, Any]]:
	return [
		{
			"name": "REFUGE2",
			"csv": root / "datasets/victorlemosml__refuge2/REFUGE2/labels.csv",
			"base_dir": root / "datasets/victorlemosml__refuge2/REFUGE2",
		},
		{
			"name": "FUNDUS_SORTED",
			"csv": root
			/ "datasets/sshikamaru__glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/labels.csv",
			"base_dir": root
			/ "datasets/sshikamaru__glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted",
		},
		{
			"name": "ACRIMA",
			"csv": root / "datasets/sshikamaru__glaucoma-detection/ACRIMA/labels.csv",
			"base_dir": root / "datasets/sshikamaru__glaucoma-detection/ACRIMA",
		},
		{
			"name": "ORIGA",
			"csv": root / "datasets/sshikamaru__glaucoma-detection/ORIGA/ORIGA/labels.csv",
			"base_dir": root / "datasets/sshikamaru__glaucoma-detection/ORIGA/ORIGA",
		},
	]


def load_one_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
	df = pd.read_csv(cfg["csv"])
	df["dataset"] = cfg["name"]

	if "split" not in df.columns:
		df["split"] = "all"
	if "image_path" not in df.columns:
		raise ValueError(f"Missing image_path in {cfg['csv']}")
	if "filename" not in df.columns:
		df["filename"] = df["image_path"].map(lambda p: Path(str(p)).name)

	if "label" in df.columns:
		df["label"] = pd.to_numeric(df["label"], errors="coerce")
	else:
		df["label"] = np.nan

	df["has_label"] = df["label"].isin([0, 1])
	df["abs_image_path"] = df["image_path"].map(lambda p: cfg["base_dir"] / str(p))
	df["image_exists"] = df["abs_image_path"].map(Path.exists)
	return df


def image_stats(path: Path) -> dict[str, float] | None:
	try:
		if cv2 is not None:
			img = cv2.imread(str(path))
			if img is None:
				return None
			h, w = img.shape[:2]
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
			blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
		else:
			# Fallback path when OpenCV is unavailable.
			rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
			h, w = rgb.shape[:2]
			gray = rgb.mean(axis=2)
			# Approximate blur score from gradient energy variance.
			gx = np.diff(gray, axis=1)
			gy = np.diff(gray, axis=0)
			blur_var = float(np.var(gx) + np.var(gy))
		return {
			"width": float(w),
			"height": float(h),
			"aspect_ratio": float(w / h) if h else np.nan,
			"brightness": float(gray.mean()),
			"contrast": float(gray.std()),
			"blur_var": blur_var,
			"red_mean": float(rgb[:, :, 0].mean()),
			"green_mean": float(rgb[:, :, 1].mean()),
			"blue_mean": float(rgb[:, :, 2].mean()),
		}
	except (UnidentifiedImageError, OSError, ValueError):
		return None


def file_md5(path: Path, chunk_size: int = 8192) -> str:
	h = hashlib.md5()
	with path.open("rb") as f:
		while True:
			block = f.read(chunk_size)
			if not block:
				break
			h.update(block)
	return h.hexdigest()


def dhash64(path: Path, hash_size: int = 8) -> int | None:
	try:
		with Image.open(path) as img:
			gray = np.array(
				img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR),
				dtype=np.uint8,
			)
		diff = gray[:, 1:] > gray[:, :-1]
		bits = "".join("1" if v else "0" for v in diff.flatten())
		return int(bits, 2)
	except (UnidentifiedImageError, OSError, ValueError):
		return None


def hamming_distance(a: int, b: int) -> int:
	return (a ^ b).bit_count()


def build_quality_flags(img_stats: pd.DataFrame) -> pd.DataFrame:
	quality_df = img_stats.copy()
	quality_df["is_blurry"] = quality_df["blur_var"] < 60
	quality_df["is_low_contrast"] = quality_df["contrast"] < 25
	quality_df["is_underexposed"] = quality_df["brightness"] < 50
	quality_df["is_overexposed"] = quality_df["brightness"] > 210
	quality_df["quality_issue_count"] = (
		quality_df[["is_blurry", "is_low_contrast", "is_underexposed", "is_overexposed"]]
		.sum(axis=1)
		.astype(int)
	)
	quality_df["quality_tier"] = pd.cut(
		quality_df["quality_issue_count"],
		bins=[-1, 0, 1, 10],
		labels=["good", "warning", "poor"],
	)
	return quality_df


def main() -> None:
	args = parse_args()
	root = args.root.resolve()
	output_dir = (args.output_dir or (root / "outputs" / "eda")).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	sns.set_theme(style="whitegrid")
	pd.set_option("display.max_columns", 200)

	csv_config = build_csv_config(root)
	for cfg in csv_config:
		if not cfg["csv"].exists():
			raise FileNotFoundError(f"Missing labels CSV: {cfg['csv']}")

	all_df = pd.concat([load_one_dataset(cfg) for cfg in csv_config], ignore_index=True)
	if args.max_images is not None and len(all_df) > args.max_images:
		all_df = all_df.sample(args.max_images, random_state=42).reset_index(drop=True)

	audit_summary = (
		all_df.groupby(["dataset", "split"], dropna=False)
		.agg(
			n_rows=("filename", "count"),
			n_labeled=("has_label", "sum"),
			n_missing_label=("has_label", lambda s: (~s).sum()),
			n_missing_image=("image_exists", lambda s: (~s).sum()),
			n_pos=("label", lambda s: (s == 1).sum()),
			n_neg=("label", lambda s: (s == 0).sum()),
		)
		.reset_index()
	)
	audit_summary["pos_ratio"] = audit_summary["n_pos"] / (
		audit_summary["n_pos"] + audit_summary["n_neg"]
	).replace(0, np.nan)

	stats_rows: list[dict[str, Any]] = []
	exists_df = all_df[all_df["image_exists"]]
	for _, row in tqdm(exists_df.iterrows(), total=len(exists_df), desc="Image stats"):
		stats = image_stats(row["abs_image_path"])
		if stats is None:
			continue
		stats_rows.append(
			{
				"dataset": row["dataset"],
				"split": row["split"],
				"filename": row["filename"],
				"label": row["label"],
				**stats,
			}
		)
	img_stats = pd.DataFrame(stats_rows)

	hash_df = all_df[all_df["image_exists"]][
		["dataset", "split", "filename", "abs_image_path"]
	].copy()
	if args.sample_hashes is not None and len(hash_df) > args.sample_hashes:
		hash_df = hash_df.sample(args.sample_hashes, random_state=42).reset_index(drop=True)
	hash_df["md5"] = [file_md5(Path(p)) for p in tqdm(hash_df["abs_image_path"], desc="Hashing")]
	duplicates = hash_df[hash_df.duplicated("md5", keep=False)].sort_values("md5")
	hash_df["dhash64"] = [dhash64(Path(p)) for p in tqdm(hash_df["abs_image_path"], desc="Perceptual hashing")]
	hash_df = hash_df[hash_df["dhash64"].notna()].copy()
	hash_df["dhash64"] = hash_df["dhash64"].astype("uint64")

	near_dups_rows: list[dict[str, Any]] = []
	if len(hash_df) >= 2:
		hashes = hash_df["dhash64"].tolist()
		for i in tqdm(range(len(hash_df) - 1), desc="Near-duplicate scan"):
			h_i = int(hashes[i])
			for j in range(i + 1, len(hash_df)):
				dist = hamming_distance(h_i, int(hashes[j]))
				if dist <= args.near_dup_distance:
					row_i = hash_df.iloc[i]
					row_j = hash_df.iloc[j]
					near_dups_rows.append(
						{
							"dataset_a": row_i["dataset"],
							"split_a": row_i["split"],
							"filename_a": row_i["filename"],
							"dataset_b": row_j["dataset"],
							"split_b": row_j["split"],
							"filename_b": row_j["filename"],
							"hamming_distance": dist,
							"same_dataset": bool(row_i["dataset"] == row_j["dataset"]),
							"same_split": bool(
								(row_i["dataset"] == row_j["dataset"])
								and (str(row_i["split"]) == str(row_j["split"]))
							),
						}
					)
	near_duplicates = pd.DataFrame(near_dups_rows)

	required_cols = {
		"dataset",
		"split",
		"image_path",
		"filename",
		"label",
		"abs_image_path",
		"image_exists",
	}
	missing_cols = required_cols - set(all_df.columns)
	if missing_cols:
		raise AssertionError(f"Missing required columns: {missing_cols}")
	if not all_df["label"].dropna().isin([0, 1]).all():
		raise AssertionError("Found labels outside {0,1}")
	if not img_stats.empty:
		if not (img_stats["width"] > 0).all() or not (img_stats["height"] > 0).all():
			raise AssertionError("Non-positive width/height found")
		if not img_stats["brightness"].between(0, 255).all():
			raise AssertionError("Brightness out of [0,255]")

	report = pd.DataFrame()
	if not img_stats.empty:
		report = (
			img_stats.groupby("dataset")
			.agg(
				n_images=("filename", "count"),
				width_median=("width", "median"),
				height_median=("height", "median"),
				brightness_median=("brightness", "median"),
				contrast_median=("contrast", "median"),
				blur_median=("blur_var", "median"),
			)
			.join(
				all_df[all_df["label"].isin([0, 1])]
				.groupby("dataset")["label"]
				.agg(n_pos=lambda s: (s == 1).sum(), n_neg=lambda s: (s == 0).sum())
			)
		)
		report["pos_ratio"] = report["n_pos"] / (report["n_pos"] + report["n_neg"])

	quality_df = pd.DataFrame()
	quality_summary = pd.DataFrame()
	if not img_stats.empty:
		quality_df = build_quality_flags(img_stats)
		quality_summary = (
			quality_df.groupby(["dataset", "split", "quality_tier"], dropna=False)
			.size()
			.rename("count")
			.reset_index()
		)

	audit_summary.to_csv(output_dir / "audit_summary.csv", index=False)
	img_stats.to_csv(output_dir / "image_stats.csv", index=False)
	duplicates.to_csv(output_dir / "duplicate_rows.csv", index=False)
	near_duplicates.to_csv(output_dir / "near_duplicate_rows.csv", index=False)
	quality_df.to_csv(output_dir / "image_quality_flags.csv", index=False)
	quality_summary.to_csv(output_dir / "image_quality_summary.csv", index=False)
	report.to_csv(output_dir / "dataset_report.csv")

	if not args.skip_plots:
		labeled_df = all_df[all_df["label"].isin([0, 1])].copy()
		labeled_df["label_name"] = labeled_df["label"].map({0: "non_glaucoma", 1: "glaucoma"})

		fig, axes = plt.subplots(1, 2, figsize=(14, 5))
		sns.countplot(data=labeled_df, x="dataset", hue="label_name", ax=axes[0])
		axes[0].set_title("Class Counts by Dataset")
		axes[0].tick_params(axis="x", rotation=20)

		ratio_df = (
			labeled_df.groupby(["dataset", "label_name"])
			.size()
			.rename("count")
			.reset_index()
		)
		ratio_df["ratio"] = ratio_df["count"] / ratio_df.groupby("dataset")["count"].transform("sum")
		sns.barplot(data=ratio_df, x="dataset", y="ratio", hue="label_name", ax=axes[1])
		axes[1].set_title("Class Ratios by Dataset")
		axes[1].tick_params(axis="x", rotation=20)
		plt.tight_layout()
		plt.savefig(output_dir / "class_distribution.png", dpi=150)
		plt.close(fig)

		if not img_stats.empty:
			fig, axes = plt.subplots(2, 2, figsize=(14, 10))
			sns.boxplot(data=img_stats, x="dataset", y="width", ax=axes[0, 0])
			axes[0, 0].set_title("Width by Dataset")
			axes[0, 0].tick_params(axis="x", rotation=20)

			sns.boxplot(data=img_stats, x="dataset", y="height", ax=axes[0, 1])
			axes[0, 1].set_title("Height by Dataset")
			axes[0, 1].tick_params(axis="x", rotation=20)

			sns.boxplot(data=img_stats, x="dataset", y="brightness", ax=axes[1, 0])
			axes[1, 0].set_title("Brightness by Dataset")
			axes[1, 0].tick_params(axis="x", rotation=20)

			sns.boxplot(data=img_stats, x="dataset", y="blur_var", ax=axes[1, 1])
			axes[1, 1].set_title("Blur Proxy (Laplacian Variance)")
			axes[1, 1].tick_params(axis="x", rotation=20)

			plt.tight_layout()
			plt.savefig(output_dir / "image_quality_boxplots.png", dpi=150)
			plt.close(fig)

			channel_plot_df = img_stats.melt(
				id_vars=["dataset", "split", "filename"],
				value_vars=["red_mean", "green_mean", "blue_mean"],
				var_name="channel",
				value_name="mean_value",
			)
			fig, ax = plt.subplots(figsize=(13, 6))
			sns.boxplot(data=channel_plot_df, x="dataset", y="mean_value", hue="channel", ax=ax)
			ax.set_title("Color Channel Shift by Dataset")
			ax.tick_params(axis="x", rotation=20)
			plt.tight_layout()
			plt.savefig(output_dir / "color_shift_by_dataset.png", dpi=150)
			plt.close(fig)

			dataset_split_color = (
				img_stats.groupby(["dataset", "split"], dropna=False)[
					["red_mean", "green_mean", "blue_mean"]
				]
				.mean()
				.reset_index()
			)
			dataset_split_color["dataset_split"] = (
				dataset_split_color["dataset"].astype(str)
				+ " | "
				+ dataset_split_color["split"].astype(str)
			)
			color_bar_df = dataset_split_color.melt(
				id_vars=["dataset_split"],
				value_vars=["red_mean", "green_mean", "blue_mean"],
				var_name="channel",
				value_name="avg_value",
			)
			fig, ax = plt.subplots(figsize=(14, 6))
			sns.barplot(data=color_bar_df, x="dataset_split", y="avg_value", hue="channel", ax=ax)
			ax.set_title("Average Color Channels by Dataset and Split")
			ax.tick_params(axis="x", rotation=35)
			plt.tight_layout()
			plt.savefig(output_dir / "color_shift_by_dataset_split.png", dpi=150)
			plt.close(fig)

			if not quality_summary.empty:
				fig, ax = plt.subplots(figsize=(12, 6))
				sns.barplot(data=quality_summary, x="dataset", y="count", hue="quality_tier", ax=ax)
				ax.set_title("Image Quality Tier Counts by Dataset")
				ax.tick_params(axis="x", rotation=20)
				plt.tight_layout()
				plt.savefig(output_dir / "quality_tiers_by_dataset.png", dpi=150)
				plt.close(fig)

	print("EDA completed")
	print(f"Rows analyzed: {len(all_df)}")
	print(f"Audit: {output_dir / 'audit_summary.csv'}")
	print(f"Image stats: {output_dir / 'image_stats.csv'}")
	print(f"Duplicates: {output_dir / 'duplicate_rows.csv'}")
	print(f"Near duplicates: {output_dir / 'near_duplicate_rows.csv'}")
	print(f"Quality flags: {output_dir / 'image_quality_flags.csv'}")
	print(f"Report: {output_dir / 'dataset_report.csv'}")


if __name__ == "__main__":
	main()
