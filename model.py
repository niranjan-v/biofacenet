import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def EncoderBlock(filters, doubleconv):
    conv_layers = []
    for i in range(len(filters)):
        if i==0:
            infilters = 3
        else:
            infilters = filters[i-1]

        conv_layers.append(
            nn.Sequential(
                nn.Conv2d(infilters, filters[i], kernel_size=3, padding=1), 
                nn.BatchNorm2d(filters[i]),
                nn.ReLU()
            ))

        if doubleconv:
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(filters[i], filters[i], kernel_size=3, padding=1), 
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.Conv2d(filters[i], filters[i], kernel_size=3, padding=1), 
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                ))

        if i<len(filters)-1:
            conv_layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

    return conv_layers

def DecoderBlock(filters, doubleconv):
    deconv_layers = []
    for i in reversed(range(len(filters)-1)):
        deconv_layers.append(nn.ConvTranspose2d(filters[i+1], filters[i+1], kernel_size=4, stride=2, padding=1))

        deconv_layers.append(
            nn.Sequential(
                nn.Conv2d(filters[i+1]+filters[i], filters[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[i]),
                nn.ReLU()
            ))
        if doubleconv:
            deconv_layers.append(
                nn.Sequential(
                    nn.Conv2d(filters[i], filters[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.Conv2d(filters[i], filters[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU()
                ))

    deconv_layers.append(nn.Conv2d(filters[0], 1, kernel_size=3, padding=1))

    return deconv_layers

class CNN(nn.Module):
    def __init__(self, nclass=4, filters=[32, 64, 128, 256, 512], doubleconv=True, LightVectorSize=15,bSize=2):
        super(CNN, self).__init__()
        self.filters = filters
        self.nclass = nclass
        self.doubleconv = doubleconv
        self.LightVectorSize = LightVectorSize
        self.bSize = bSize
        self.fcdim = LightVectorSize + bSize

        self.encoder = nn.Sequential(*EncoderBlock(self.filters, self.doubleconv))      
        self.decoders = nn.ModuleList([nn.Sequential(*DecoderBlock(self.filters, self.doubleconv)) for _ in range(self.nclass)])
        self.fc = nn.Sequential(
                        nn.Conv2d(filters[-1], filters[-1], kernel_size=4),
                        nn.BatchNorm2d(filters[-1]),
                        nn.ReLU(),
                        nn.Conv2d(filters[-1], filters[-1], kernel_size=1),
                        nn.BatchNorm2d(filters[-1]),
                        nn.ReLU(),
                        nn.Conv2d(filters[-1], self.fcdim, kernel_size=1)
                    )

    def forward(self, x):
        conv_feats = []
        for m in self.encoder.children():
            classname = m.__class__.__name__
            if classname.find('MaxPool2d') != -1:
                conv_feats.append(x)
            x = m(x)
        
        for dnum, d in enumerate(self.decoders):
            cidx = -1
            xd = x 
            for m in d.children():    
                xd = m(xd)            
                classname = m.__class__.__name__
                if classname.find('ConvTranspose2d') != -1:
                    xd=torch.cat([xd,conv_feats[cidx]],1)
                    cidx = cidx -1

            if dnum==0:
                z=xd
            else:
                z=torch.cat([z,xd],1) # [N,C,H,W]

        return x,z

    def predict(self, x):
        y,z = self.forward(x)
        icpar = self.fc(y) # illumination, camera params - [N,17,1,1]
        lightingparameters = icpar[:,0:self.LightVectorSize].squeeze() # [N,15,1,1] -> [N,15]
        b = icpar[:,self.LightVectorSize:].squeeze() # [N,2,1,1] -> [N, 2] 

        fmel = z[:,0]
        fblood = z[:,1]
        Shading = z[:,2]
        specmask = z[:,3] # [N,H,W]

        return lightingparameters,b,fmel,fblood,Shading,specmask
