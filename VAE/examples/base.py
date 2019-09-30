import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from collections import defaultdict
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.autograd import Variable
import os
import pandas as pd
import itertools
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import globalvar as gl
import copy

#######################################################################################
########################################加载参数########################################
use_cuda=gl.get_value('use_cuda')
device=gl.get_value('device')
log_interval=gl.get_value('log_interval')
log_dir=gl.get_value('log_dir')
lr=gl.get_value('lr')
batch_size=gl.get_value('batch_size')
n_cpu=gl.get_value('n_cpu')
latent_dim=gl.get_value('latent_dim')
h_dim=gl.get_value('h_dim')
img_size=gl.get_value('img_size')
distribution=gl.get_value('distribution')
C_VAE=gl.get_value('C_VAE')

writer = SummaryWriter(gl.get_value('log_dir'))
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################################################################
########################################网络结构########################################
def reparameterization(z_mean, z_var):
    # q_z是网络估计的概率值,p_z是希望逼近的概率值
    if distribution == 'normal':
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
    elif distribution == 'vmf':
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(latent_dim - 1)
    else:
        raise NotImplemented

    return q_z, p_z

class Encoder(nn.Module):
    def __init__(self,distribution='normal'):
        super(Encoder,self).__init__()
        self.distribution=distribution
        if C_VAE==True:
            self.BN=nn.BatchNorm1d(img_size**2+1)
            self.e0 = nn.Linear(img_size**2+1, h_dim * 2)
        else:
            self.BN=nn.BatchNorm1d(img_size**2)
            self.e0 = nn.Linear(img_size**2, h_dim * 2)
        self.activation = nn.ReLU()
        self.e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.mean = nn.Linear(h_dim, latent_dim)
            self.var = nn.Linear(h_dim, latent_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.mean = nn.Linear(h_dim, latent_dim)
            self.var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented

    def forward(self,img):
        img_flat = img.view(img.shape[0], -1)
        
        x = self.activation(self.e0(img_flat))
        x = self.activation(self.e1(x))

        if self.distribution=='normal':
            z_mean=self.mean(x)
            z_var=nn.Softplus()(self.var(x))
        elif self.distribution=='vmf':
            z_mean=self.mean(x)
            z_mean=z_mean/z_mean.norm(dim=-1,keepdim=True)
            z_var=nn.Softplus()(self.var(x))
        else:
            raise NotImplemented
        q_z,p_z = reparameterization(z_mean, z_var)
        z=q_z.rsample()
        return z,(q_z,p_z)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        if C_VAE==True:
            self.d0=nn.Linear(latent_dim+1,h_dim)
        else:
            self.d0=nn.Linear(latent_dim,h_dim)

        self.d1=nn.Linear(h_dim,h_dim*2)
        self.logits=nn.Linear(h_dim*2,img_size**2)
        self.activation = nn.ReLU()

    def forward(self, z):
        x=self.activation(self.d0(z))
        x=self.activation(self.d1(x))
        x=nn.Tanh()(self.logits(x))
        return x

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
             transforms.ToTensor()]
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
             transforms.ToTensor()]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# def getidx(points,mx,MX,my,MY,step):
#     n_col=int((MX-mx)/step)
#     idx=np.zeros((points.shape[0],1)).astype(np.int)
#     for i in range(0,points.shape[0]):
#         idx[i]=int((MY-points[i][1])/step)*n_col+int((points[i][0]-mx)/step)
#     return idx


# def sample2d(decoder,points,idx,c, path,name,n_row,n_col):
#     """Saves a grid of generated digits"""
#     data=points
#     if C_VAE==True:
#         data=np.hstack((data,c))
#     z = Variable(Tensor(data))
#     gen_imgs = decoder(z).view(-1,1,28,28)
#     save_img=torch.zeros((n_row*n_col,1,28,28)).to('cuda')
#     if  os.path.exists(path)==False:
#         os.makedirs(path)

#     for i in range(0,gen_imgs.shape[0]):
#         pdb.set_trace()
#         save_img[idx[i,0],0,:,:]=gen_imgs[i,0,:,:]
    
#     save_image(save_img, os.path.join(path,name),
#                nrow=n_row, normalize=True)

def sample2d(decoder,point,c, path,name):
    data=point
    if C_VAE==True:
        data=np.hstack((data,c))
    z = Variable(Tensor(data))
    gen_imgs = decoder(z).view(-1,1,28,28)
    
    writer=SummaryWriter(path)
    writer.add_embedding(point,metadata=None,label_img=gen_imgs,tag=name)

#######################################################################################
########################################实例化网络######################################
class VAE():
    def __init__(self):
        self.encoder=Encoder(distribution)
        self.decoder=Decoder()
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()
        self.optimizer = optim.Adam(itertools.chain(self.encoder.parameters(),
                                                    self.decoder.parameters()),
                                    lr=lr)

    def train(self,s_epoch,e_epoch,lossKL=True):
        if s_epoch>0:
            self.load(log_dir,s_epoch)
        
        globalstep = s_epoch * len(dataloader) 
        for epoch in range(s_epoch,e_epoch):
            for i, (imgs, y) in enumerate(dataloader):
                real_imgs = Variable(imgs.type(Tensor))
                real_imgs=(real_imgs>torch.distributions.Uniform(0, 1)
                           .sample(real_imgs.shape).to(device)).float().to(device)
                y=Variable(y.view(-1,1).type(Tensor))
                #-------------------------
                #   train
                #-------------------------
                self.optimizer.zero_grad()
                if C_VAE==True:
                	inputdata=torch.cat([real_imgs.view(real_imgs.shape[0],-1),y],1)
                else:
                	inputdata=real_imgs.view(real_imgs.shape[0],-1)
                encoded_imgs,(q_z,p_z)=self.encoder(inputdata)
                if C_VAE==True:
                	inputdata2=torch.cat([encoded_imgs,y],1)
                else:
                	inputdata2=encoded_imgs
                decoded_imgs=self.decoder(inputdata2)
                #loss value
                loss_recon = nn.BCEWithLogitsLoss(reduction='none')\
                    (decoded_imgs, real_imgs.reshape(-1, 784)).sum(-1).mean()
                loss_KL=0

                if lossKL==True:
                    if distribution == 'normal':
                        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
                        pdb.set_trace()
                    elif distribution == 'vmf':
                        loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
                    else:
                        raise NotImplemented
                else:
                    pass
                
                loss = loss_recon + loss_KL

                # if torch.isnan(loss):
                #     pdb.set_trace()
                globalstep = globalstep + 1
                loss.backward()
                self.optimizer.step()

                if distribution=='normal':
                    writer.add_scalars('nVAE',
                                       {'loss_recon': loss_recon, 'loss_KL': loss_KL, 'loss': loss},
                                       global_step=globalstep)

                if distribution=='vmf':
                    writer.add_scalars('sVAE',
                                       {'loss_recon': loss_recon, 'loss_KL': loss_KL, 'loss': loss},
                                       global_step=globalstep)

            if distribution == 'normal':
                writer.add_scalars('nVAE',{'epoch':1},global_step=epoch)

            if distribution == 'vmf':
                writer.add_scalars('sVAE',{'epoch':1},global_step=epoch)

    def save(self,param_dir,e_epoch):
        if distribution == 'normal':
            torch.save(self.encoder.state_dict(),
                       os.path.join(param_dir,
                                    'nVAE/{}d_encoder_{}epoch.pkl'.format(latent_dim,e_epoch))
                       )
            torch.save(self.decoder.state_dict(),
                       os.path.join(param_dir,
                                    'nVAE/{}d_decoder_{}epoch.pkl'.format(latent_dim, e_epoch))
                       )

        if distribution=='vmf':
            torch.save(self.encoder.state_dict(),
                       os.path.join(param_dir,
                                    'sVAE/{}d_encoder_{}epoch.pkl'.format(latent_dim, e_epoch))
                       )
            torch.save(self.decoder.state_dict(),
                       os.path.join(param_dir,
                                    'sVAE/{}d_decoder_{}epoch.pkl'.format(latent_dim, e_epoch))
                       )

    def load(self,param_dir,e_epoch):
        if distribution == 'normal':
            self.encoder.load_state_dict(torch.load(
                os.path.join(param_dir,
                             'nVAE/{}d_encoder_{}epoch.pkl'.format(latent_dim,e_epoch))
            ))
            self.decoder.load_state_dict(torch.load(
                os.path.join(param_dir,
                             'nVAE/{}d_decoder_{}epoch.pkl'.format(latent_dim, e_epoch))
            ))

        if distribution == 'vmf':
            self.encoder.load_state_dict(torch.load(
                os.path.join(param_dir,
                             'sVAE/{}d_encoder_{}epoch.pkl'.format(latent_dim,e_epoch))
            ))
            self.decoder.load_state_dict(torch.load(
                os.path.join(param_dir,
                             'sVAE/{}d_decoder_{}epoch.pkl'.format(latent_dim, e_epoch))
            ))

    def generate_hidden(self,log_dir):
        hcode = np.zeros((60000, latent_dim + 1))
        idx = 0
        for i, (imgs, y) in enumerate(dataloader):
            real_imgs= Variable(imgs.type(Tensor))
            y=Variable(y.view(-1,1).type(Tensor))
            if C_VAE==True:
                inputdata=torch.cat([real_imgs.view(real_imgs.shape[0],-1),y],1)
            else:
                inputdata=real_imgs
            encoded_imgs,(q_z,p_z)  = self.encoder(inputdata)
            hcode[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            hcode[idx:idx + imgs.shape[0], latent_dim] = y.cpu().detach().numpy()[:,0]
            idx += imgs.shape[0]

        df = pd.DataFrame(hcode)
        if distribution=='normal':
            df.to_csv(os.path.join(log_dir, 'nVAE/{}d.csv'.format(latent_dim)), index=False)
        if distribution=='vmf':
            df.to_csv(os.path.join(log_dir, 'sVAE/{}d.csv'.format(latent_dim)), index=False)

        print('data has been saved successfully')

    def plot(self):
        if latent_dim==2:
            if distribution=='normal':
                df=pd.read_csv(os.path.join(log_dir,'nVAE/{}d.csv'.format(latent_dim)))
                plt.scatter(df.iloc[:,0],df.iloc[:,1],c=df.iloc[:,2])
                plt.show()

            if distribution=='vmf':
                df = pd.read_csv(os.path.join(log_dir, 'sVAE/{}d.csv'.format(latent_dim)))
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:,2])
                plt.show()
        if latent_dim==3:
            if distribution=='normal':
                df=pd.read_csv(os.path.join(log_dir,'nVAE/{}d.csv'.format(latent_dim)))
                fig=plt.figure()
                ax=Axes3D(fig)
                ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=df.iloc[:,3])
                plt.show()

            if distribution=='vmf':
                df=pd.read_csv(os.path.join(log_dir,'sVAE/{}d.csv'.format(latent_dim)))
                fig=plt.figure()
                ax=Axes3D(fig)
                ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=df.iloc[:,3])
                plt.show()
    
    def pred(self,n_neighbors,weights,plot):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

        #--------准备训练数据
        train_code = np.zeros((60000, latent_dim+1))
        idx = 0
        for i, (imgs, y) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            encoded_imgs,(q_z,p_z) = self.encoder(real_imgs)
            train_code[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            train_code[idx:idx + imgs.shape[0], latent_dim] = y
            idx += imgs.shape[0]

        #----------准备test数据
        test_code = np.zeros((10000, latent_dim+1))
        idx = 0
        for i, (imgs, y) in enumerate(testloader):
            real_imgs = Variable(imgs.type(Tensor))
            encoded_imgs,(q_z,p_z) = self.encoder(real_imgs)
            test_code[idx:idx + imgs.shape[0], 0:latent_dim] = encoded_imgs.cpu().detach().numpy()
            test_code[idx:idx + imgs.shape[0], latent_dim] = y
            idx += imgs.shape[0]

        clf.fit(train_code[:,0:latent_dim],train_code[:,latent_dim])
        y_pred=clf.predict(test_code[:,0:latent_dim])
        acc=accuracy_score(test_code[:,latent_dim],y_pred)
        print('accuracy is {}'.format(acc))

        pd1=pd.DataFrame(train_code)
        pd2=pd.DataFrame(test_code)
        pred_code=copy.copy(test_code)
        pred_code[:,latent_dim]=y_pred
        pd3=pd.DataFrame(pred_code)
        pd1.to_csv(os.path.join(log_dir,'images/train_code{}d_{}.csv'.format(latent_dim,distribution)))
        pd2.to_csv(os.path.join(log_dir,'images/test_code{}d_{}.csv'.format(latent_dim,distribution)))
        pd3.to_csv(os.path.join(log_dir,'images/pred_code{}d_{}.csv'.format(latent_dim,distribution)))
        
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
                plot.subplot(212)
                ax.scatter(test_code[:,0],test_code[:,1],test_code[:,2],c=test_code[:,3])
                plt.show()
        return acc
