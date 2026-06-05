"""
Script to plot one image from each blur level from JRAIGS dataset + histogram
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
import numpy as np

# Define base paths
DATASET_PATH = "datasets/jraigs_prepared"
LABELS_CSV = os.path.join(DATASET_PATH, "labels.csv")
IMAGES_DIR = os.path.join(DATASET_PATH, "images")

def load_images_by_blur():
    """Load the labels and group images by blur ranges"""
    df = pd.read_csv(LABELS_CSV)
    
    # Remove any NaN blur scores
    df = df.dropna(subset=['blur_score'])
    
    # Create dynamic blur ranges based on the data
    min_blur = df['blur_score'].min()
    max_blur = df['blur_score'].max()
    
    # Create bins every 50 blur units (or 20 bins total)
    n_bins = min(20, max(5, int((max_blur - min_blur) / 50)))
    bins = np.linspace(min_blur, max_blur, n_bins + 1)
    
    print(f"Blur range in dataset: {min_blur:.1f} to {max_blur:.1f}")
    print(f"Creating {len(bins)-1} blur bins")
    
    # Create a dictionary to store one representative image per blur range
    blur_images = {}
    blur_ranges = []
    
    for i in range(len(bins) - 1):
        blur_min = bins[i]
        blur_max = bins[i + 1]
        label = f"{blur_min:.0f}-{blur_max:.0f}"
        blur_ranges.append((blur_min, blur_max, label))
        
        # Filter images in this blur range
        mask = (df['blur_score'] >= blur_min) & (df['blur_score'] < blur_max)
        images_in_range = df[mask]
        
        if len(images_in_range) > 0:
            # Pick a random image from this range
            chosen_row = images_in_range.sample(1).iloc[0]
            image_name = chosen_row['kept_image'].split('/')[-1]
            image_path = os.path.join(IMAGES_DIR, image_name)
            
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    blur_value = chosen_row['blur_score']
                    blur_images[label] = {
                        'image': img,
                        'blur': blur_value,
                        'path': image_path
                    }
                except:
                    print(f"Failed to load {image_path}")
    
    return blur_images, blur_ranges, df

def get_blur_color(blur_value, min_blur, max_blur):
    """Get color based on blur value (low blur = green, high blur = red)"""
    normalized = (blur_value - min_blur) / (max_blur - min_blur + 1e-6)
    # Green (low blur) to Red (high blur)
    return (normalized, 1 - normalized, 0)

def create_blur_plot_and_histogram(output_path_plot="blur_levels.png", output_path_hist="blur_histogram.png"):
    """Create a plot with images from each blur range + histogram"""
    
    # Load images
    blur_images, blur_ranges, df = load_images_by_blur()
    n_ranges = len(blur_images)
    
    if n_ranges == 0:
        print("No images found!")
        return
    
    min_blur = df['blur_score'].min()
    max_blur = df['blur_score'].max()
    
    # ===== PART 1: Image Grid =====
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (n_ranges + n_cols - 1) // n_cols
    
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Load and display images
    loaded_count = 0
    for idx, (range_label, data) in enumerate(blur_images.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        img = data['image']
        blur_val = data['blur']
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        
        # Create title with blur value
        ax.set_title(f"{range_label}\n(~{blur_val:.1f})", 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border color based on blur (green = sharp, red = blurry)
        color = get_blur_color(blur_val, min_blur, max_blur)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        loaded_count += 1
    
    print(f"Loaded {loaded_count} images from {n_ranges} blur ranges")
    
    # Create legend with blur gradient (green to red)
    legend_elements = []
    for blur_min, blur_max, label in blur_ranges:
        mid_blur = (blur_min + blur_max) / 2
        color = get_blur_color(mid_blur, min_blur, max_blur)
        legend_elements.append(
            mpatches.Patch(facecolor=color, 
                          edgecolor='black',
                          label=f"{label}")
        )
    
    fig.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(0.87, 0.5),
              fontsize=9,
              title='Blur Score Ranges',
              title_fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True,
              ncol=1)
    
    fig.suptitle('JRAIGS Images by Blur Level (Green=Sharp, Red=Blurry)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path_plot, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path_plot}")
    plt.close()
    
    # ===== PART 2: Histogram =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram
    n_hist_bins = 50
    counts, bins, patches = ax.hist(df['blur_score'], bins=n_hist_bins, edgecolor='black', alpha=0.7)
    
    # Color bars from green (low blur) to red (high blur)
    for i, patch in enumerate(patches):
        bin_mid = (bins[i] + bins[i+1]) / 2
        color = get_blur_color(bin_mid, min_blur, max_blur)
        patch.set_facecolor(color)
    
    ax.set_xlabel('Blur Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Blur Scores in JRAIGS Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_blur = df['blur_score'].mean()
    median_blur = df['blur_score'].median()
    std_blur = df['blur_score'].std()
    
    stats_text = f"Mean: {mean_blur:.1f}\nMedian: {median_blur:.1f}\nStd: {std_blur:.1f}\nN: {len(df)}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path_hist, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram to {output_path_hist}")
    plt.close()

if __name__ == "__main__":
    create_blur_plot_and_histogram()
