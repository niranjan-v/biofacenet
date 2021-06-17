import torch
import torch.nn.functional as F
import scipy.io as io
import torch.nn as nn
import time, os, sys, math
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from CustomTensorDataset import *
from model import *
from helpers import *
from Utils import *

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', action='store',  required=True, help='test data file path')
    parser.add_argument('--model', action='store', required=True, help='model weights file path')

    args = parser.parse_args()

    data_path = args.data
    model_path = args.model

    data =  torch.load(data_path).float()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    u = Utils('./util/',device)
    model = CNN(nclass=4, filters=[32, 64, 128, 256, 512], doubleconv=True, LightVectorSize=u.LightVectorSize,bSize=u.bSize).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    start_idx=100 #run data[start_idx:start_idx+batch_sz] through the model
    batch_sz=64
    images1=data[start_idx:start_idx+batch_sz,0:3]
    images=(data[start_idx:start_idx+batch_sz,0:3].to(device))**2.2 - u.muim.view(-1,1,1) #invert gamma correction and centre data to pass on as i/p to network
    actualmasks=data[start_idx:start_idx+batch_sz,3].to(device)
    actualshading=data[start_idx:start_idx+batch_sz,4].to(device)

    with torch.no_grad():
        lightingparameters,b,fmel,fblood,predictedShading,specmask = model.predict(images)
        weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,predictedShading,specmask = scalingNet(lightingparameters,b,fmel,fblood,predictedShading,specmask,u.bSize)
        e = illuminationModel(weightA,weightD,Fweights,CCT,u.illumA,u.illumDNorm,u.illumFNorm)
        Sr,Sg,Sb=cameraModel(u.mu,u.PC,b,u.wavelength)
        lightcolour = computelightcolour(e,Sr,Sg,Sb)
        Specularities = computeSpecularities(specmask,lightcolour)
        R_total = BiotoSpectralRef(fmel,fblood,u.Newskincolour) #BiotoSpectralRef(0.5+fmel,fblood,u.Newskincolour) - to see the effect of increasing melanin by 0.5
        rawAppearance,diffuseAlbedo = ImageFormation(R_total, Sr,Sg,Sb,e,Specularities,predictedShading)
        ImwhiteBalanced = WhiteBalance(rawAppearance,lightcolour)
        T_RAW2XYZ = findT(u.Tmatrix,BGrid)
        sRGBim = fromRawTosRGB(ImwhiteBalanced,T_RAW2XYZ,u.Txyzrgb)

## display results

    inums = [30,40] #pick a few indices to display 
    for i in inums:
        fig, axs = plt.subplots(3,2)

        axs[0,0].set_title("actual img")
        axs[0,0].imshow((images1[i]**actualmasks[i].cpu()).permute(1,2,0).numpy())
        axs[0,0].axis('off')
        
        axs[0,1].set_title("reconstructed img")
        axs[0,1].imshow((sRGBim[i]**actualmasks[i]).permute(1,2,0).detach().cpu().numpy()**(1/2.2))
        axs[0,1].axis('off')
  
        axs[1,0].set_title("melanin map")
        axs[1,0].imshow(cm.jet(0.5*(1+fmel[i]).detach().cpu().numpy()) * actualmasks[i].unsqueeze(-1).cpu().numpy())
        axs[1,0].axis('off')

        axs[1,1].set_title("haemoglobin map")
        axs[1,1].imshow(cm.hot(0.5*(1+fblood[i]).detach().cpu().numpy()) * actualmasks[i].unsqueeze(-1).cpu().numpy())
        axs[1,1].axis('off')

        axs[2,0].set_title("shading map")
        axs[2,0].imshow(cstretch(predictedShading[i].detach().cpu().numpy()) ** actualmasks[i].cpu().numpy(), cmap='gray')
        axs[2,0].axis('off')

        axs[2,1].set_title("specularities map")
        axs[2,1].imshow(cstretch(specmask[i].detach().cpu().numpy()) ** actualmasks[i].cpu().numpy(), cmap='gray')
        axs[2,1].axis('off')
        
        plt.show()
