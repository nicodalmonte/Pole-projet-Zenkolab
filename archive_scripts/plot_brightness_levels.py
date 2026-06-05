"""
Script to plot one image from each brightness level range from JRAIGS dataset
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

# Define brightness ranges - will be populated dynamically
BRIGHTNESS_RANGES = []

# Color gradient for brightness levels
def get_brightness_color(brightness_value):
    """Get color based on brightness value (0 = black, 100 = white)"""
    normalized = brightness_value / 100.0
    return plt.cm.gray(normalized)

def load_images_by_brightness():
    """Load the labels and group images by brightness ranges"""
    df = pd.read_csv(LABELS_CSV)
    
    # Create dynamic brightness ranges based on the data
    min_brightness = df['brightness'].min()
    max_brightness = df['brightness'].max()
    
    # Create bins every 5 brightness units
    bin_width = 5
    bins = np.arange(int(min_brightness), int(max_brightness) + bin_width, bin_width)
    
    print(f"Brightness range in dataset: {min_brightness:.1f} to {max_brightness:.1f}")
    print(f"Creating {len(bins)-1} brightness bins with width={bin_width}")
    
    # Create a dictionary to store one representative image per brightness range
    brightness_images = {}
    brightness_ranges = []
    
    for i in range(len(bins) - 1):
        brightness_min = bins[i]
        brightness_max = bins[i + 1]
        label = f"{brightness_min:.0f}-{brightness_max:.0f}"
        brightness_ranges.append((brightness_min, brightness_max, label))
        
        # Filter images in this brightness range
        mask = (df['brightness'] >= brightness_min) & (df['brightness'] < brightness_max)
        images_in_range = df[mask]
        
        if len(images_in_range) > 0:
            # Pick a random image from this range
            chosen_row = images_in_range.sample(1).iloc[0]
            image_name = chosen_row['kept_image'].split('/')[-1]
            image_path = os.path.join(IMAGES_DIR, image_name)
            
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    brightness_value = chosen_row['brightness']
                    brightness_images[label] = {
                        'image': img,
                        'brightness': brightness_value,
                        'path': image_path
                    }
                except:
                    print(f"Failed to load {image_path}")
    
    return brightness_images, brightness_ranges

def create_brightness_plot(output_path="brightness_levels.png"):
    """Create a plot with images from each brightness range"""
    
    # Load images
    brightness_images, brightness_ranges = load_images_by_brightness()
    n_ranges = len(brightness_images)
    
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
    for idx, (range_label, data) in enumerate(brightness_images.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        img = data['image']
        brightness_val = data['brightness']
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        
        # Create title with brightness value
        ax.set_title(f"{range_label}\n(~{brightness_val:.1f})", 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border color based on brightness
        color = get_brightness_color(brightness_val)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        loaded_count += 1
    
    print(f"Loaded {loaded_count} images from {n_ranges} brightness ranges")
    
    # Create legend with brightness gradient
    legend_elements = []
    for brightness_min, brightness_max, label in brightness_ranges:
        mid_brightness = (brightness_min + brightness_max) / 2
        color = get_brightness_color(mid_brightness)
        legend_elements.append(
            mpatches.Patch(facecolor=color, 
                          edgecolor='black',
                          label=f"{label}")
        )
    
    fig.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(0.87, 0.5),
              fontsize=9,
              title='Brightness Ranges',
              title_fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True,
              ncol=1)
    
    fig.suptitle('JRAIGS Images by Brightness Level (Full Range)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_brightness_plot()
