import torch
import torch.nn


class LAGDataset(torch.utils.data.dataset):
    def __init__(self,root_dir,split):
        super.__init__()
        self.root_dir=root_dir
        self.split=split
        self.samples=[]

    def __len__(self):
        return(len(self.samples))
    def __getitem__(self,idx):
        picture,label=self.samples[idx]
        return 