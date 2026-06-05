"""
Script to plot one image from each color level from JRAIGS dataset
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
import numpy as np
from colorsys import rgb_to_hsv

# Define base paths
DATASET_PATH = "datasets/jraigs_prepared"
LABELS_CSV = os.path.join(DATASET_PATH, "labels.csv")
IMAGES_DIR = os.path.join(DATASET_PATH, "images")

def parse_color(color_str):
    """Parse RGB string like '77.22,57.34,35.26' to normalized RGB tuple"""
    try:
        r, g, b = map(float, color_str.split(','))
        return (r / 255.0, g / 255.0, b / 255.0)
    except:
        return None

def get_color_metric(rgb_tuple):
    """Get luminance (brightness) from RGB - for sorting colors"""
    r, g, b = rgb_tuple
    return 0.299 * r + 0.587 * g + 0.114 * b

def load_images_by_color():
    """Load the labels and group images by color ranges"""
    df = pd.read_csv(LABELS_CSV)
    
    # Parse RGB colors
    df['rgb_tuple'] = df['color_mean_rgb'].apply(parse_color)
    df = df.dropna(subset=['rgb_tuple'])
    
    # Get luminance for sorting
    df['luminance'] = df['rgb_tuple'].apply(get_color_metric)
    
    # Create dynamic color ranges based on luminance
    min_luminance = df['luminance'].min()
    max_luminance = df['luminance'].max()
    
    # Create bins every 0.05 luminance units
    bin_width = 0.05
    bins = np.arange(min_luminance, max_luminance + bin_width, bin_width)
    
    print(f"Luminance range in dataset: {min_luminance:.3f} to {max_luminance:.3f}")
    print(f"Creating {len(bins)-1} color bins with width={bin_width}")
    
    # Create a dictionary to store one representative image per color range
    color_images = {}
    color_ranges = []
    
    for i in range(len(bins) - 1):
        lum_min = bins[i]
        lum_max = bins[i + 1]
        label = f"{lum_min:.2f}-{lum_max:.2f}"
        color_ranges.append((lum_min, lum_max, label))
        
        # Filter images in this luminance range
        mask = (df['luminance'] >= lum_min) & (df['luminance'] < lum_max)
        images_in_range = df[mask]
        
        if len(images_in_range) > 0:
            # Pick a random image from this range
            chosen_row = images_in_range.sample(1).iloc[0]
            image_name = chosen_row['kept_image'].split('/')[-1]
            image_path = os.path.join(IMAGES_DIR, image_name)
            
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    rgb_value = chosen_row['rgb_tuple']
                    luminance = chosen_row['luminance']
                    color_images[label] = {
                        'image': img,
                        'rgb': rgb_value,
                        'luminance': luminance,
                        'path': image_path
                    }
                except:
                    print(f"Failed to load {image_path}")
    
    return color_images, color_ranges

def create_color_plot(output_path="color_levels.png"):
    """Create a plot with images from each color range"""
    
    # Load images
    color_images, color_ranges = load_images_by_color()
    n_ranges = len(color_images)
    
    if n_ranges == 0:
        print("No images found!")
        return
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (n_ranges + n_cols - 1) // n_cols
    
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Load and display images
    loaded_count = 0
    for idx, (range_label, data) in enumerate(color_images.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        img = data['image']
        rgb_val = data['rgb']
        luminance = data['luminance']
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        
        # Create title with color info
        ax.set_title(f"Luminance: {range_label}\n(RGB: {rgb_val[0]:.2f}, {rgb_val[1]:.2f}, {rgb_val[2]:.2f})", 
                    fontsize=9, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border color based on actual RGB
        for spine in ax.spines.values():
            spine.set_edgecolor(rgb_val)
            spine.set_linewidth(3)
        
        loaded_count += 1
    
    print(f"Loaded {loaded_count} images from {n_ranges} color ranges")
    
    # Create legend with color gradient
    legend_elements = []
    for lum_min, lum_max, label in color_ranges:
        mid_lum = (lum_min + lum_max) / 2
        # Create a color based on luminance (grayscale)
        color = (mid_lum, mid_lum, mid_lum)
        legend_elements.append(
            mpatches.Patch(facecolor=color, 
                          edgecolor='black',
                          label=f"{label}")
        )
    
    fig.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(0.87, 0.5),
              fontsize=8,
              title='Luminance Ranges',
              title_fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              ncol=1)
    
    fig.suptitle('JRAIGS Images by Color (Luminance)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_color_plot()
