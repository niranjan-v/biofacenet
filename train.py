import torch
import torch.nn.functional as F
import scipy.io as io
import torch.nn as nn
import time, os, sys, math
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from CustomTensorDataset import *
from model import *
from helpers import *
from Utils import *
import numpy as np
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tdata', action='store',  required=True, help='training data file path')
    parser.add_argument('--vdata', action='store', required=True, help='validation data file path')

    args = parser.parse_args()
    tdata_path = args.tdata
    vdata_path = args.vdata

    tdata =  torch.load(tdata_path).float()
    vdata =  torch.load(vdata_path).float()

    train_d = CustomTensorDataset(tensors=(tdata[:,0:3], tdata[:,3:]), transform=linimg)
    train_loader = utils.DataLoader(train_d, batch_size=64, shuffle=True, num_workers=1)

    val_d = CustomTensorDataset(tensors=(vdata[:,0:3], vdata[:,3:]), transform=linimg)
    val_loader = utils.DataLoader(val_d, batch_size=64, shuffle=True, num_workers=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    u = Utils('./util/',device)
    model = CNN(nclass=4, filters=[32, 64, 128, 256, 512], doubleconv=True, LightVectorSize=u.LightVectorSize,bSize=u.bSize).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    epochs = 200
    train_losses = []
    val_losses = []
    lepoch = -1 #last epoch
    cpt_path = './checkpoint.pt'

    '''
    use this to load checkpoint if training is interrupted and start from last epoch + 1
    checkpoint = torch.load(cpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lepoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    '''

    for epoch in range(1+lepoch, epochs):
        model.train()
        train_loss = 0.0
        split_loss = torch.zeros(4)
        start = time.time()
        for batch_idx, (images, preds) in enumerate(train_loader):       
            images=images.to(device)
            actualmasks=preds[:,0].to(device)
            actualshading=preds[:,1].to(device)
            #
            optimizer.zero_grad()
            lightingparameters,b,fmel,fblood,predictedShading,specmask = model.predict(images)
            loss, loss_s = calc_loss(images,actualmasks,actualshading,lightingparameters,b,fmel,fblood,predictedShading,specmask,u)        
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            split_loss += loss_s
        train_losses.append(train_loss)
        print("Epoch {} time elapsed {:.2f} sec training loss {} loss breakdown {}".format(epoch+1, time.time()-start, train_loss, split_loss.tolist()))

        if (epoch+1)%5==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses' : val_losses
                        }, cpt_path)
            print("saved checkpoint at epoch", epoch+1)
        
        if epoch%5==3:
            #validation
            model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = 0.0
                split_loss = torch.zeros(4)
                for batch_idx, (images, preds) in enumerate(val_loader):       
                    images=images.to(device)
                    actualmasks=preds[:,0].to(device)
                    actualshading=preds[:,1].to(device)
                    #
                    lightingparameters,b,fmel,fblood,predictedShading,specmask = model.predict(images)
                    loss, loss_s = calc_loss(images,actualmasks,actualshading,lightingparameters,b,fmel,fblood,predictedShading,specmask,u)
                    split_loss += loss_s
                    val_loss += loss.item()
                val_losses.append(val_loss)
            print("Epoch {} time elapsed {:.2f} sec validation loss {} loss breakdown {}".format(epoch+1, time.time()-start, val_loss, split_loss.tolist()))

    plt.figure()
    plt.plot([i+1 for i in range(len(train_losses))] ,train_losses)
    plt.title("training loss")
    plt.show()

    plt.figure()
    plt.plot([5*i+4 for i in range(len(val_losses))] ,val_losses)
    plt.title("validation loss")
    plt.show()