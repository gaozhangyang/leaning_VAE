##保存60000个训练数据的三维隐变量至csv

from sphere_VAE import ModelVAE
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter


H_DIM = 128
Z_DIM = 3
seed=1
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = SummaryWriter(log_dir='./logs2d')

hcode_N=np.zeros((60000,4))
hcode_S=np.zeros((60000,4))

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=60000, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=10000)


################################################################################################
#-------------------------------for normal and sphere VAE---------------------------------------
# normal VAE
modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal').to(device)
optimizerN = optim.Adam(modelN.parameters(), lr=1e-3)

# hyper-spherical  VAE
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM , distribution='vmf').to(device)
optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)



modelN.load_state_dict(torch.load('./result2d/Normal_epoch{}_step{}.pkl'.format(49,(49+1)*938)))
modelN.eval()

modelS.load_state_dict(torch.load('./result2d/Hsphere_epoch{}_step{}.pkl'.format(49, (49 + 1) * 938)))
modelS.eval()

for i, (x_mb, y_mb) in enumerate(train_loader):
    x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float().to(device)

    _, (q_z_N, p_z_N), ZN, x_mb_N = modelN(x_mb.reshape(-1, 784))
    hcode_N[:,0:3]=ZN.cpu().detach().numpy()
    hcode_N[:,3]=y_mb

    _, (q_z_S, p_z_S), ZS, x_mb_S = modelS(x_mb.reshape(-1, 784))
    hcode_S[:,0:3]=ZS.cpu().detach().numpy()
    hcode_S[:,3]=y_mb

dfN=pd.DataFrame(hcode_N)
dfS=pd.DataFrame(hcode_S)
dfN.to_csv('./result2d/Normal.csv',index=False)
dfS.to_csv('./result2d/Sphere.csv',index=False)



################################################################################################
#-------------------------if only use loss_KL, what will happen?--------------------------------
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM , distribution='vmf').to(device)
modelS.load_state_dict(torch.load('./result2d/sphere_direct_KL_epoch{}.pkl'.format(2)))
modelS.eval()

for i, (x_mb, y_mb) in enumerate(train_loader):
    x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float().to(device)

    _, (q_z_S, p_z_S), ZS, x_mb_S = modelS(x_mb.reshape(-1, 784))
    hcode_S[:,0:3]=ZS.cpu().detach().numpy()
    hcode_S[:,3]=y_mb

dfS=pd.DataFrame(hcode_S)
dfS.to_csv('./result2d/Sphere_direct_KL_epoch2.csv',index=False)