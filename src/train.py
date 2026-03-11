from src.data.dataset import LAGDataset
from src.data.transforms import get_train_transforms, get_val_transforms
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


training_set= LAGDataset(root= "datasets/LAG/LAG/train/", transform=get_train_transforms())
validation_set= LAGDataset(root= "datasets/LAG/LAG/validation/", transform=get_val_transforms())
test_set= LAGDataset(root= "datasets/LAG/LAG/test/", transform=get_val_transforms())

