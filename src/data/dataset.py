from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class LAGDataset(Dataset):

    def __init__(self, root, transform=None):
        # 1. salva root e transform come attributi dell'oggetto
        self.root=Path(root) #root viene trasformato in path
        self.transform= transform 
        # 2. trova tutti i file .jpg in root e salvali in una lista
        self.images = sorted(self.root.glob("*.jpg"))
    def __len__(self):
        # restituisce quante immagini ci sono in totale
        return len(self.images)
    def __getitem__(self, idx):
        # 1. prendi il path dell'immagine numero idx dalla tua lista
        img_path = self.images[idx]
        # 2. apri l'immagine con PIL
        image = Image.open(img_path).convert("RGB")
        # 3. ricava il label dal nome del file
        #    g → 1, ng → 0
        prefix = img_path.stem.split(".")[0]
        label = 1 if prefix == "g" else 0
        # se prefix=="g"  → label = 1 (malato)
        # se prefix=="ng" → label = 0 (sano)
        # 4. applica il transform se non è None
        if self.transform is not None:
            image = self.transform(image)
        
        # 5. Restituisci la coppia (immagine, label)
        return image, label
        # 5. restituisci immagine e label