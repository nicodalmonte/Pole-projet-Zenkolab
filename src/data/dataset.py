from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class GlaucomaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform 
        self.images = sorted(self.root.glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # g -> 1 (glaucoma), altrimenti -> 0 (sano)
        prefix = img_path.stem.split(".")[0]
        label = 1 if prefix == "g" else 0
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label