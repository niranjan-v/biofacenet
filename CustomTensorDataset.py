import torch
import torch.utils.data as utils
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.models as models

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def linimg(img):
    #invert gamma and adjust mean
    muim = torch.tensor([129.1863,104.7624,93.5940])/255
    imgLinear = img**2.2 - muim.view(-1,1,1)
    return imgLinear
