import torch
import torch.nn as nn
import timm
import torchvisions

def createhead(embedding_dim,num_classes,dropout_rate):
    return(nn.Sequential(nn.Layernorm(),nn.Dropout(),nn.Linear(embedding_dim,num_classes)))

class VitModel(nn.module):
    def __init__(self,embeddings,nums_classes,size_picture):
       super().__init__
       self.embeddings=embeddings
       self.nums_classes=nums_classes
       self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
       self.size_picture=size_picture
       self.head=createhead()

    def adapt_picture(size_picture,picture):
        return(picture.torchvision.transform.Resize([size_picture,size_picture]))
    def forward(x):
        x=adapt_picture(size_picture,x)
        x=self.backbone(x)
        x=self.head(x)
        return(x)


    

