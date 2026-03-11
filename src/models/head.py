from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512,2)
        )
    def forward(self,x):
        return self.layers(x)
    

