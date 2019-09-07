import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from collections import defaultdict
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from measure import measure
import matplotlib.pyplot as plt

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

seed = 1
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda:0" if use_cuda else "cpu")
globalstep = 0
log_interval = 100
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                                          transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
                                                         transform=transforms.ToTensor()), batch_size=10000)
writer = SummaryWriter(log_dir='./logs2d')


class ModelVAE(torch.nn.Module):

    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()

        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented

        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented

        return z_mean, z_var

    def decode(self, z):

        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        return x

    def reparameterize(self, z_mean, z_var,y):
        # q_z是网络估计的概率值,p_z是希望逼近的概率值
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.tensor(y), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z=VonMisesFisher(z_mean, z_var)
        p_z=None
        z = q_z.rsample()
        x_ = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, x_

direct=np.zeros((10,3))
idx=0
for a in range(0,2):
    for b in range(0,5):
        if idx==10:
            break
        direct[idx,0]=np.cos(a/2*2*np.pi)*np.cos(b/5*2*np.pi)
        direct[idx, 1] = np.sin(a/2*2*np.pi)*np.cos(b/5*2*np.pi)
        direct[idx, 2] = np.sin(b / 5 * 2 * np.pi)
        idx=idx+1
direct=torch.tensor(direct,dtype=torch.float32).to(device)

def train(model, optimizer, epoch, globalstep, method, device):
    for i, (x_mb, y_mb) in enumerate(train_loader):
        means = torch.zeros(x_mb.shape[0], 3).to(device)
        vars = torch.zeros(x_mb.shape[0], 1).to(device)
        optimizer.zero_grad()

        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float().to(device)

        _, (q_z, p_z), z, x_mb_ = model(x_mb.reshape(-1, 784))
        p_z=torch.zeros(x_mb.shape[0],3,dtype=torch.float32).to(device)
        for idx in range(0,x_mb.shape[0]):
            means[idx,:]=direct[y_mb[idx],:]
            vars[idx,:]=30
        p_z=VonMisesFisher(means, vars)

        loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 784)).sum(-1).mean()

        if model.distribution == 'normal':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif model.distribution == 'vmf':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplemented

        loss = 0*loss_recon + 1000*loss_KL
        writer.add_scalars('{} scalars'.format(method), {'loss_recon': loss_recon, 'loss_KL': loss_KL, 'loss': loss},globalstep)
        globalstep = globalstep + 1
        loss.backward()
        optimizer.step()
    return globalstep


if __name__ == '__main__':
    # hidden dimension and dimension of latent space
    H_DIM = 128
    Z_DIM = 3

    # hyper-spherical  VAE
    modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='vmf').to(device)
    optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)

    print('##### Hyper-spherical VAE #####')
    globalstep = 0
    for epoch in range(0, 50):
        # training for 1 epoch
        globalstep = train(modelS, optimizerS, epoch, globalstep, 'sphere direct KL ', device)
        # with torch.no_grad():
        #     test(modelS, optimizerS, globalstep, 'Hsphere VAE', device)
        print(epoch)
    torch.save(modelS.state_dict(), './result2d/sphere_direct_KL_epoch{}.pkl'.format(epoch))


