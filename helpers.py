import torch
import torch.nn as nn
import torch.nn.functional as F

def scalingNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize=2):
    '''
    Inputs:
        lightingparameters              :   [N,15]
        b                               :   [N,2]  
        fmel,fblood,Shading,specmask    :   [N,H,W]
    Outputs:
        weightA, weightD                :   [N]
        CCT                             :   [N]
        Fweights                        :   [N,12]
        b, BGrid                        :   [N,2]  
        fmel,fblood,Shading,specmask    :   [N,H,W]
    '''
    lightingweights = F.softmax(lightingparameters[:,0:14],1) # [N,14]
    weightA = lightingweights[:,0] # [N]
    weightD = lightingweights[:,1]
    Fweights = lightingweights[:,2:14] # [N,12]
    CCT = lightingparameters[:,14] # [N]
    CCT = (21/(1 + torch.exp(-CCT))) # scale [0-21] since 0 indexing
    #
    b = 6*torch.sigmoid(b) - 3
    BGrid =  b/3.0  #

    fmel = 2*torch.sigmoid(fmel)-1
    fblood = 2*torch.sigmoid(fblood)-1
    Shading = torch.exp(Shading)
    specmask = torch.exp(specmask)

    return weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,Shading,specmask


def illuminationModel(weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm):
    '''
    Inputs:
        weightA, weightD                :   [N]
        Fweights                        :   [N,12]
        CCT                             :   [N]
        illumA                          :   [33]
        illumDNorm                      :   [33,22]
        illumFNorm                      :   [33,12]
    Output:
        e                               :   [N,33]
    '''
    N = weightA.numel()
    
# illumination A:
    illuminantA = weightA.view(-1,1)*illumA.view(1,-1) # [N,33] = [N,1]*[1,33]

# illumination D:
    illumDNorm = illumDNorm.view(1,1,33,22)
    grid = torch.zeros(N,1,33,2).to(illumDNorm.device)
    grid[:,0,:,0] = (2*CCT/21-1).view(N,1)
    grid[:,0,:,1]=torch.tensor([-1+i*(2/32) for i in range(33)]) # [N,33] <- [33]
    illumD = F.grid_sample(illumDNorm.expand(N,-1,-1,-1), grid, align_corners=True).squeeze() # [N,1,1,33] -> [N,33]
    illuminantD = weightD.view(-1,1) * illumD # [N,33] = [N,1] * [N,33]

# illumination F:
    illumFNorm = illumFNorm.view(33,12,1)
    Fweights = Fweights.permute(1,0) # [12,N]
    illuminantF = illumFNorm*Fweights 
    illuminantF = illuminantF.sum(1).permute(1,0) # [33,12,N] -> [33,N] -> [N,33]

    e = illuminantA + illuminantD + illuminantF 
    e = e/e.sum(1,keepdim=True)

    return e


def cameraModel(mu,PC,b,wavelength):
    '''
    Inputs:
        mu         :  [99,1]
        PC         :  [99,2]
        b          :  [N,2] 
    Outputs:
        Sr,Sg,Sb   :  [N,33]
    '''
    N = b.shape[1]
    # PCA model
    S = torch.matmul(PC,b.T) + mu # [99,N]
    S =  F.relu(S) 

    Sr = S[0:wavelength].permute(1,0)  # [33,N] -> [N,33]           
    Sg = S[wavelength:wavelength*2].permute(1,0)
    Sb = S[wavelength*2:wavelength*3].permute(1,0)

    return Sr,Sg, Sb


def computelightcolour(e,Sr,Sg,Sb):
    '''
    Inputs:
        Sr,Sg,Sb         : [N,33]
        e                : [N,33]
    Output:
        lightcolour      : [N,3]
    '''
    lightcolour = torch.stack([(Sr*e).sum(1), (Sg*e).sum(1), (Sb*e).sum(1)],1)

    return lightcolour


def computeSpecularities(specmask,lightcolour):
    '''
    Inputs:
        specmask         : [N,H,W]
        lightcolour      : [N,3]
    Output:
        Specularities    : [N,3,H,W]
    '''
    specmask = specmask.unsqueeze(1) # [N,1,H,W]
    Specularities = specmask*lightcolour.view(-1,3,1,1) # [N,3,H ,W] = [N,1,H,W ] * [N,3,1,1]

    return Specularities


def BiotoSpectralRef(fmel,fblood,Newskincolour):
    '''
    Inputs:
        fmel,fblood      : [N,H,W]
        Newskincolour    : [1,33,256,256]
    Output:
        R_total          : [N,33,H,W]
    '''
    N = fmel.shape[0]
    BiophysicalMaps =   torch.cat([fmel.unsqueeze(1),fblood.unsqueeze(1)], dim=1) # [N,2,H,W]   
    BiophysicalMaps =   BiophysicalMaps.permute(0,2,3,1)  # [N,H,W,2], grid
    Newskincolour = Newskincolour.expand(N,-1,-1,-1) # [N,33,256,256]
    R_total  = F.grid_sample(Newskincolour, BiophysicalMaps, align_corners=True)
    return R_total


def ImageFormation (R_total, Sr,Sg,Sb,e,Specularities,Shading):
    '''
    Inputs:
        R_total          : [N,33,H,W]
        Sr,Sg,Sb,e       : [N,33]
        Specularities    : [N,3,H,W]
        Shading          : [N,H,W]
    Output:
        rawAppearance, diffuseAlbedo : [N,3,H,W]
    ''' 
    # N = e.shape[0]
    spectraRef = R_total*e.view(-1,33,1,1) # N x 33 x H x W
    rChannel = (spectraRef*Sr.view(-1,33,1,1)).sum(1,keepdim=True)
    gChannel = (spectraRef*Sg.view(-1,33,1,1)).sum(1,keepdim=True)
    bChannel = (spectraRef*Sb.view(-1,33,1,1)).sum(1,keepdim=True) # [N,1,H,W]

    diffuseAlbedo = torch.cat([rChannel,gChannel,bChannel], dim=1) # [N,3,H,W]
    ShadedDiffuse = diffuseAlbedo*(Shading.unsqueeze(1)) # [N,3,H,W] = [N,3,H,W] * [N,1,H,W]
    rawAppearance = ShadedDiffuse + Specularities

    return rawAppearance, diffuseAlbedo


def WhiteBalance(rawAppearance,lightcolour):
    '''
    Inputs:
        rawAppearance    : [N,3,H,W]
        lightcolour      : [N,3]
    Output:
        ImwhiteBalanced  : [N,3,H,W]
    '''
    ImwhiteBalanced = rawAppearance/lightcolour.view(-1,3,1,1) # [N,3,H,W] = [N,3,H,W] / [N,3,1,1]

    return ImwhiteBalanced


def findT(Tmatrix,BGrid):
    '''
    Inputs:
        Tmatrix          : [1,9,128,128]
        BGrid            : [N,2]
    Output:
        T_RAW2XYZ        : [N,9]
    '''
    N = BGrid.shape[0]
    T_RAW2XYZ = F.grid_sample(Tmatrix.expand(N,-1,-1,-1), BGrid.view(N,1,1,2), align_corners=True).squeeze() # [N,9,1,1] -> [N,9]

    return T_RAW2XYZ


def fromRawTosRGB(imWB,T_RAW2XYZ,Txyzrgb):
    '''
    Inputs:
        imWB             : [N,3,H,W]
        T_RAW2XYZ        : [N,9]
        Txyzrgb          : [N,9]
    Output:
        T_RAW2XYZ        : [N,9]
    '''
    T_R2X = T_RAW2XYZ.view(-1,3,3,1,1) # [[1,2,3],[4,5,6],[7,8,9]]
    Ix = (T_R2X[:,:,0] * imWB).sum(1) # [N,3,1,1] * [N,3,H,W] -> [N,3,H,W] -> [N,H,W]
    Iy = (T_R2X[:,:,1] * imWB).sum(1)
    Iz = (T_R2X[:,:,2] * imWB).sum(1)

    Ixyz = torch.stack([Ix,Iy,Iz],1) # [N,3,H,W]

    R = (Txyzrgb[0,:].view(-1,1,1) * Ixyz).sum(1) # [3,1,1] * [N,3,H,W] -> [N,3,H,W] -> [N,H,W]
    G = (Txyzrgb[1,:].view(-1,1,1) * Ixyz).sum(1)
    B = (Txyzrgb[2,:].view(-1,1,1) * Ixyz).sum(1)

    sRGBim = torch.stack([R,G,B],1) # [N,3,H,W]
    sRGBim =F.relu(sRGBim)

    return sRGBim


def priorLoss(b, weight):
    loss = (b*b).sum()
    return loss*weight


def appearanceLoss(rgbim, images, muim, actualmasks, weight):
    rgb = rgbim - muim.view(-1,1,1)
    delta = (images - rgb)*(actualmasks.unsqueeze(1))
    loss = (delta*delta).sum()
    return loss*weight


def sparsityLoss(Specularities, weight):
    #L1 sparsity loss
    loss = Specularities.sum()
    return loss*weight


def shadingLoss(actualshading, predictedShading, actualmasks, weight):
    '''
    Inputs:
        actualshading, predictedShading : [N,H,W]
        actualmasks                     : [N,H,W] 
    '''
    scale = ((actualshading*predictedShading)*actualmasks).sum(2).sum(1)/((predictedShading**2)*actualmasks).sum(2).sum(1)  # [N]

    predictedShading = predictedShading*scale.view(-1,1,1)
    alpha = (actualshading - predictedShading)*actualmasks
    loss = (alpha**2).sum()
    return loss*weight


def calc_loss(images,actualmasks,actualshading,lightingparameters,b,fmel,fblood,predictedShading,specmask,u):
    '''
    fn to compute losses during training, validation - for modularizing the training code
    '''

    weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,predictedShading,specmask = scalingNet(lightingparameters,b,fmel,fblood,predictedShading,specmask,u.bSize)
    e = illuminationModel(weightA,weightD,Fweights,CCT,u.illumA,u.illumDNorm,u.illumFNorm)
    Sr,Sg,Sb=cameraModel(u.mu,u.PC,b,u.wavelength)
    lightcolour = computelightcolour(e,Sr,Sg,Sb)
    Specularities = computeSpecularities(specmask,lightcolour)
    R_total = BiotoSpectralRef(fmel,fblood,u.Newskincolour)
    rawAppearance,diffuseAlbedo = ImageFormation(R_total, Sr,Sg,Sb,e,Specularities,predictedShading)
    ImwhiteBalanced = WhiteBalance(rawAppearance,lightcolour)
    T_RAW2XYZ = findT(u.Tmatrix,BGrid)
    sRGBim = fromRawTosRGB(ImwhiteBalanced,T_RAW2XYZ,u.Txyzrgb)

    prloss = priorLoss(b, u.blossweight)
    aploss = appearanceLoss(sRGBim, images, u.muim, actualmasks, u.appweight)
    sploss = sparsityLoss(Specularities, u.sparseweight)
    shloss = shadingLoss(actualshading, predictedShading, actualmasks, u.Shadingweight)

    loss = prloss + aploss + sploss + shloss
    loss_s = torch.tensor([aploss.item(), shloss.item(), prloss.item(), sploss.item()]) #loss breakdown

    return loss, loss_s
