import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
import pandas as pd
import globalvar as gl
import pdb

#######################################################################################
########################################加载参数########################################
writer=SummaryWriter(gl.get_value('log_dir'))
batch_size=gl.get_value('batch_size')
lr=gl.get_value('lr')
b1=gl.get_value('b1')
b2=gl.get_value('b2')
n_cpu=gl.get_value('n_cpu')
h_dim=gl.get_value('h_dim')
latent_dim=gl.get_value('latent_dim')
img_size=gl.get_value('img_size')
channels=gl.get_value('channels')
sample_interval=gl.get_value('sample_interval')
img_shape=gl.get_value('img_shape')
cuda=gl.get_value('cuda')
log_dir=gl.get_value('log_dir')
C_VAE=gl.get_value('C_VAE')

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#######################################################################################
########################################网络结构########################################
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), h_dim*2),
            nn.ReLU(),
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(h_dim, latent_dim)
        self.logvar = nn.Linear(h_dim, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, h_dim*2),
            nn.ReLU(),
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, h_dim*2),
            nn.ReLU(),
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


#######################################################################################
######################################tool function####################################
# Configure data loader
os.makedirs("../data", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

testloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

def sample2d(decoder,xscope,yscope,c, path,name):
    """Saves a grid of generated digits"""
    xnums=np.arange(*xscope)
    ynums=np.arange(*yscope)
    L=len(xnums)*len(ynums)
    x,y=np.meshgrid(xnums,ynums)
    data=np.hstack((np.reshape(x,(L,1)),np.reshape(y,(L,1))))
    if C_VAE==True:
        data=np.hstack((data,c))
    n_row=len(ynums)
    z = Variable(Tensor(data))
    gen_imgs = decoder(z)
    if  os.path.exists(path)==False:
        os.makedirs(path)
    gen_imgs=gen_imgs.view(L,1,28,28)
    save_image(gen_imgs.data, os.path.join(path,name),
               nrow=n_row, normalize=True)

#######################################################################################
########################################实例化网络######################################
class AAE():
    def __init__(self):
        # Use binary cross-entropy loss
        self.adversarial_loss = torch.nn.BCELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        if cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                                            self.decoder.parameters()),
                                            lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=lr, betas=(b1, b2))

    def train(self,s_epoch,e_epoch):
        if s_epoch>0:
            self.load(log_dir,s_epoch)

        global_step = s_epoch * len(dataloader) 
        for epoch in range(s_epoch,e_epoch):
            for i, (imgs, y) in enumerate(dataloader):
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001*self.adversarial_loss(self.discriminator(encoded_imgs), valid)\
                         + 0.999 * self.pixelwise_loss(decoded_imgs, real_imgs)
                writer.add_scalars('AAE', {'g_loss': g_loss}, global_step=global_step)
                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                writer.add_scalars('AAE',
                                   {'real_loss': real_loss, 'fake_loss': fake_loss, 'd_loss': d_loss},
                                   global_step=global_step)
                d_loss.backward()
                self.optimizer_D.step()
                global_step = global_step + 1
            writer.add_scalars('AAE',
                               {'epoch': 1},
                               global_step=epoch)

    def save(self,param_dir,e_epoch):
        torch.save(self.encoder.state_dict(),
                   os.path.join(param_dir,
                                'AAE/{}d_encoder_{}epoch.pkl'.format(latent_dim,e_epoch)
                                )
                   )
        torch.save(self.decoder.state_dict(),
                   os.path.join(param_dir,
                                'AAE/{}d_decoder_{}epoch.pkl'.format(latent_dim,e_epoch)
                                )
                   )
        torch.save(self.discriminator.state_dict(),
                   os.path.join(param_dir,
                                'AAE/{}d_discriminator_{}epoch.pkl'.format(latent_dim,e_epoch)
                                )
                   )

    def load(self,param_dir,e_epoch):
        self.encoder.load_state_dict(torch.load(
            os.path.join(param_dir,
                         'AAE/{}d_encoder_{}epoch.pkl'.format(latent_dim,e_epoch))
        ))
        self.decoder.load_state_dict(torch.load(
            os.path.join(param_dir,
                         'AAE/{}d_decoder_{}epoch.pkl'.format(latent_dim,e_epoch))
        ))
        self.discriminator.load_state_dict(torch.load(
            os.path.join(param_dir,
                         'AAE/{}d_discriminator_{}epoch.pkl'.format(latent_dim,e_epoch))
        ))

    def generate_hidden(self,log_dir):
        hcode = np.zeros((60000, latent_dim+1))
        idx = 0
        for i, (imgs, y) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            encoded_imgs = self.encoder(real_imgs)
            hcode[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            hcode[idx:idx + imgs.shape[0], latent_dim] = y
            idx += imgs.shape[0]
        df = pd.DataFrame(hcode)
        df.to_csv(os.path.join(log_dir,'AAE/{}d.csv'.format(latent_dim)), index=False)
        print('data has been saved successfully')
    
    def plot(self):
        if latent_dim==2:
            df=pd.read_csv(os.path.join(log_dir,'AAE/{}d.csv'.format(latent_dim)))
            plt.scatter(df.iloc[:,0],df.iloc[:,1],c=df.iloc[:,2])
            plt.show()

        if latent_dim==3:
            df=pd.read_csv(os.path.join(log_dir,'AAE/{}d.csv'.format(latent_dim)))
            fig=plt.figure()
            ax=Axes3D(fig)
            ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=df.iloc[:,3])
            plt.show()
    
    def pred(self,n_neighbors,weights,plot):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

        #--------准备训练数据
        train_code = np.zeros((60000, latent_dim+1))
        idx = 0
        for i, (imgs, y) in enumerate(testloader):
            real_imgs = Variable(imgs.type(Tensor))
            encoded_imgs = self.encoder(real_imgs)
            train_code[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            train_code[idx:idx + imgs.shape[0], latent_dim] = y
            idx += imgs.shape[0]

        #----------准备test数据
        test_code = np.zeros((10000, latent_dim+1))
        idx = 0
        for i, (imgs, y) in enumerate(testloader):
            real_imgs = Variable(imgs.type(Tensor))
            encoded_imgs = self.encoder(real_imgs)
            test_code[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            test_code[idx:idx + imgs.shape[0], latent_dim] = y
            idx += imgs.shape[0]

        clf.fit(train_code[:,0:latent_dim],train_code[:,latent_dim])
        y_pred=clf.predict(test_code[:,0:latent_dim])
        acc=accuracy_score(test_code[:,latent_dim],y_pred)
        print('accuracy is {}'.format(acc))
        
        if plot==True:
            if latent_dim==2:
                plt.subplot(211)
                plt.scatter(train_code[:,0],train_code[:,1],c=train_code[:,2])
                plt.subplot(212)
                plt.scatter(test_code[:,0],test_code[:,1],c=test_code[:,2])
                plt.show()
            if latent_dim==3:
                fig=plt.figure()
                ax=Axes3D(fig)
                plt.subplot(211)
                ax.scatter(train_code[:,0],train_code[:,1],train_code[:,2],c=train_code[:,3])
                plt.subplot(212)
                ax.scatter(test_code[:,0],test_code[:,1],test_code[:,2],c=test_code[:,3])
                plt.show()
        return acc

            

