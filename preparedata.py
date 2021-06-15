import h5py
import numpy as np
import io, os, sys
import torch

import argparse

c1 = 0.429043 
c2 = 0.511664
c3 = 0.743125
c4 = 0.886227
c5 = 0.247708
def KMat(L):
    '''
    see equation(5) in equation (4) from https://arxiv.org/pdf/1704.04131.pdf 
    '''
    r0=np.array([c1*L[8], c1*L[4], c1*L[7], c2*L[3] ])
    r1=np.array([c1*L[4], -c1*L[8], c1*L[5], c2*L[1] ])
    r2=np.array([c1*L[7], c1*L[5], c3*L[6], c2*L[2]])
    r3=np.array([c2*L[3], c2*L[1], c2*L[2], c4*L[0]-c5*L[6]])
    K=np.array([r0,r1,r2,r3])
    return K

def Shading(img, L):
    '''
    generate shading map from normals and spherical harmonics
    implementation of equation (4) from https://arxiv.org/pdf/1704.04131.pdf 
    '''
    #img H x W x 3
    K=KMat(L) # 4 x 4
    imgp=np.concatenate([img,np.ones([img.shape[0], img.shape[1], 1])],2) # H x W x 4
    imgp=np.sum( np.matmul(imgp,K) * imgp,2) # H x W
    #imgp=imgp/imgp.max()
    return imgp # H x W

def rgb2gray(rgb):
    gimg = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    c = np.min(gimg)
    d = np.max(gimg)
    gimg = (gimg-c)/(d-c)    
    return gimg

'''
download files from : https://drive.google.com/drive/folders/1UMiaw36z2E1F-tUBSMKNAjpx0o2TePvF
for eg, to prepare test data, download zx_7_d10_inmc_celebA_20.hdf5 and zx_7_d3_lrgb_celebA_20.hdf5
to genearete data, run 
    python --inmc <path>/zx_7_d10_inmc_celebA_20.hdf5 --lrgb <path>/zx_7_d3_lrgb_celebA_20.hdf5

similarly, download celebA_01 to celebA_05 files and prepare data for training
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inmc', action='store',  required=True, help='inmc file path')
    parser.add_argument('--lrgb', action='store', required=True, help='lrgb weights file path')

    args = parser.parse_args()

    inmc_file = args.inmc
    lrgb_file = args.lrgb


    inmc_hf = h5py.File(inmc_file, 'r')
    lrgb_hf = h5py.File(lrgb_file, 'r')
    inmc = np.array(inmc_hf['zx_7'])
    lrgb = np.array(lrgb_hf['zx_7'])
    # print(inmc.shape, lrgb.shape)

    ### prepare data bin ###
    ds=inmc.shape[0]

    data = np.zeros([ds,5,64,64])
    data[:,0:3]=inmc[0:ds,0:3] #img
    data[:,3]=inmc[0:ds,6] #mask

    for inum in range(ds):
        s=np.array([Shading(inmc[inum,3:6].transpose([1,2,0]), lrgb[inum,lnum]) for lnum in range(3)])
        data[inum,4]=rgb2gray(s.transpose([1,2,0])) #shading

    '''
    data structure:
        data[:,0:3] - image
        data[:,3]   - mask
        data[:,4]   - shading map 
    '''
    data=torch.from_numpy(data).float()
    torch.save(data,'./celebA_20.bin')

    ##---------------------------------------------------------------------------------------------------------------------------------------------##