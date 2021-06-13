import torch
import scipy.io as io

'''
class to load util data
'''
class Utils():
    def __init__(self,path,device):
        self.LightVectorSize = 15
        self.wavelength = 33
        self.bSize = 2
        self.blossweight = 1e-4  
        self.appweight = 1e-3
        self.Shadingweight = 1e-5 
        self.sparseweight = 1e-5

        #load data
        Newskincolour = io.loadmat(path+'Newskincolour.mat')
        Newskincolour = torch.tensor(Newskincolour['Newskincolour'])
        Newskincolour = Newskincolour.permute(2,0,1).unsqueeze(0) # [1,33,256,256]
        self.Newskincolour = Newskincolour.to(device)
        
        Tmatrix = io.loadmat(path+'Tmatrix.mat')
        Tmatrix=torch.tensor(Tmatrix['Tmatrix'])
        Tmatrix = Tmatrix.permute(2,0,1).unsqueeze(0)
        self.Tmatrix = Tmatrix.to(device)
        
        illD = io.loadmat(path+'illumDmeasured.mat')
        illumDmeasured = torch.tensor(illD['illumDmeasured']) 
        illumDmeasured = illumDmeasured.permute(1,0)
        illumDNorm = illumDmeasured/illumDmeasured.sum(0,keepdim=True) # [33,22]
        self.illumDNorm = illumDNorm.to(device)

        illA=io.loadmat(path+'illumA.mat')
        illumA=torch.tensor(illA['illumA']).squeeze() #[1,1,33] -> [33]
        illumA = illumA/illumA.sum()
        self.illumA = illumA.to(device)


        illF=io.loadmat(path+'illF.mat')
        illumF=torch.tensor(illF['illF']) #[1,33,12]
        illumF = illumF.squeeze() # [33,12]
        illumFNorm = illumF/illumF.sum(0,keepdim=True)
        self.illumFNorm = illumFNorm.to(device)

        self.Txyzrgb = torch.tensor([[3.2406, -1.5372, -0.4986], 
                        [-0.9689, 1.8758, 0.0415], 
                        [0.0557, -0.2040, 1.057]]).to(device)

        self.muim = torch.tensor([0.5394,0.4184,0.3569])

        data=io.loadmat(path+'CamPCA.mat')
        self.EV=torch.tensor(data['EV']).to(device)
        self.PC=torch.tensor(data['PC']).to(device)
        self.mu=torch.tensor(data['mu']).to(device)

