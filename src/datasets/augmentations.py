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
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    ])
    return augmentation_transforms


# Expose the augmentation transforms as a configurable module
AUGMENTATION_TRANSFORMS = create_augmentation_transforms()
