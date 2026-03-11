from torch import nn
from src.models.backbone import build_backbone
from src.models.head import ClassificationHead
class DINOv3large(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone= build_backbone()
        self.head=ClassificationHead      #li registro come due registri separati
    
    def forward(self,x):
        return self.head(self.backbone(x))
        



