"""Data augmentation transforms for training.

Exposes a Sequential module of augmentation transforms that can be easily modified.
Uses torchvision.transforms for flexibility and consistency.
"""

from torchvision import transforms


def create_augmentation_transforms():
    """Create and return a Sequential of augmentation transforms.
    
    Returns:
        transforms.Sequential: Composed augmentation transforms for training data.
                              Can be easily modified by adding/removing transforms.
    """
    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=15),  # Random rotation ±15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random translation ±10%
            scale=(0.9, 1.1),  # Random scaling 90%-110%
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
        #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Gaussian blur
        #transforms.ColorJitter(
        #    brightness=0.2,
        #    contrast=0.2,
        #    saturation=0.2,
        #),  # Random color jittering
    ])
    return augmentation_transforms


# Expose the augmentation transforms as a configurable module
AUGMENTATION_TRANSFORMS = create_augmentation_transforms()
