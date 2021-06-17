import torch, h5py
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

    ### prepare data
    sub_h5 = h5py.File(data_path, 'r')
    skeys = list(sub_h5.keys()) #['dataset_1', 'mask', 'ppg'] 

    images_n = np.array(sub_h5[skeys[0]])/255
    masks_n = np.array(sub_h5[skeys[1]])
    # ppg_n = np.array(sub_h5[skeys[2]])

    nframes = 64 
    images = torch.from_numpy(images_n[0:nframes]).float() #test first  'nframes' frames in the video
    masks = torch.from_numpy(masks_n[0:nframes]).float()

    m_aap =  nn.AdaptiveAvgPool2d((64,64)) #use adaptive avg pooling to downsample image to 64 x 64
    m_mxp = nn.MaxPool2d(2,2) #use max pooling to downsample image to 64 x 64
    images = m_aap(images[0:64].permute(0,3,1,2))
    masks = m_mxp(masks[0:64].permute(0,3,1,2))

    ###

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    u = Utils('./util/',device)
    model = CNN(nclass=4, filters=[32, 64, 128, 256, 512], doubleconv=True, LightVectorSize=u.LightVectorSize,bSize=u.bSize).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()


    images_lc=(images.to(device))**2.2 - u.muim.view(-1,1,1) #invert gamma correction and centre the data
    actualmasks=masks.squeeze().to(device)

    with torch.no_grad():
        lightingparameters,b,fmel,fblood,predictedShading,specmask = model.predict(images_lc)
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
    fs = 12 # figure size in subplot

    for i in range(2): #edit this to choose which imgs to display
        fig, axs = plt.subplots(1,6, figsize=(fs,fs))

        # axs[0,0].set_title("actual img")
        axs[0].imshow((images[i]**actualmasks[i].cpu()).permute(1,2,0).numpy())
        axs[0].axis('off')
        
        # axs[0,1].set_title("reconstructed img")
        axs[5].imshow((sRGBim[i]**actualmasks[i]).permute(1,2,0).detach().cpu().numpy()**(1/2.2))
        axs[5].axis('off')
  
        # axs[1,0].set_title("melanin map")
        axs[1].imshow(cm.jet(0.5*(1+fmel[i]).detach().cpu().numpy()) * actualmasks[i].unsqueeze(-1).cpu().numpy())
        axs[1].axis('off')

        # axs[1,1].set_title("haemoglobin map")
        axs[2].imshow(cm.hot(0.5*(1+fblood[i]).detach().cpu().numpy()) * actualmasks[i].unsqueeze(-1).cpu().numpy())
        axs[2].axis('off')

        # axs[2,0].set_title("shading map")
        axs[3].imshow(cstretch(predictedShading[i].detach().cpu().numpy()) ** actualmasks[i].cpu().numpy(), cmap='gray')
        axs[3].axis('off')

        # axs[2,1].set_title("specularities map")
        axs[4].imshow(cstretch(specmask[i].detach().cpu().numpy()) ** actualmasks[i].cpu().numpy(), cmap='gray')
        axs[4].axis('off')
        
        plt.show()
