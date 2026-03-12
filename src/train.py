from src.data.dataset import LAGDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
import torch
from src.models.model import DINOv3large
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

training_set= LAGDataset(root= "datasets/LAG/LAG/train/", transform=get_train_transforms())
validation_set= LAGDataset(root= "datasets/LAG/LAG/validation/", transform=get_val_transforms())
test_set= LAGDataset(root= "datasets/LAG/LAG/test/", transform=get_val_transforms())

train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
validation_dataloader= DataLoader(validation_set, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)    

model= DINOv3large().to(device)