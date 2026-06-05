"""
Script to plot one image from each JRAIGS characteristic subset with legend
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
from pathlib import Path

# Define base path and subsets
BASE_PATH = "datasets/jraigs_subsets"
SUBSETS = [
    "brightness_high", "brightness_low", "brightness_medium", 
    "brightness_over_90", "brightness_under_20"
]

# Create categorical grouping for better organization
CATEGORIES = {
    "Brightness": ["brightness_high", "brightness_low", "brightness_medium", "brightness_over_90", "brightness_under_20"]
}

# Color mapping for categories
CATEGORY_COLORS = {
    "Brightness": "#4ECDC4"
}

def get_first_image(subset_name):
    """Get the first image from a subset"""
    images_path = os.path.join(BASE_PATH, subset_name, "images")
    if os.path.exists(images_path):
        images = sorted([f for f in os.listdir(images_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if images:
            img_path = os.path.join(images_path, images[0])
            return Image.open(img_path)
    return None

def create_characteristic_plot(output_path="characteristic_subsets.png"):
    """Create a plot with images from each characteristic subset"""
    
    # Calculate grid dimensions
    n_subsets = len(SUBSETS)
    n_cols = 5
    n_rows = (n_subsets + n_cols - 1) // n_cols
    
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3, 
                  right=0.85)  # Leave space for legend
    
    # Load and display images
    loaded_count = 0
    for idx, subset_name in enumerate(SUBSETS):
        row = idx // n_cols
        col = idx % n_cols
        
        img = get_first_image(subset_name)
        if img is not None:
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            
            # Find which category this subset belongs to
            category = None
            for cat, subsets in CATEGORIES.items():
                if subset_name in subsets:
                    category = cat
                    break
            
            # Create title with category color
            ax.set_title(subset_name, fontsize=11, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add border color based on category
            if category:
                color = CATEGORY_COLORS[category]
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
            
            loaded_count += 1
    
    print(f"Loaded {loaded_count} images from {n_subsets} subsets")
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat], 
                      edgecolor='black', 
                      label=f"{cat} ({len(subsets)} subsets)")
        for cat, subsets in CATEGORIES.items()
    ]
    
    fig.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(0.87, 0.5),
              fontsize=12,
              title='Characteristic Categories',
              title_fontsize=13,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    fig.suptitle('JRAIGS Characteristic Subsets - Sample Images', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_characteristic_plot()
