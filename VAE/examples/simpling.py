from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import numpy as np
from normal_VAE import *

H_DIM = 128
Z_DIM = 3
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal').to(device)
modelN.load_state_dict(torch.load('./result2d/Normal_epoch49_step46900.pkl'))
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM , distribution='vmf').to(device)
modelS.load_state_dict(torch.load('./result2d/Hsphere_epoch49_step46900.pkl'))

def sampleN_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, Z_DIM))))
    gen_imgs = modelN.decode(z)
    gen_imgs=nn.Tanh()(gen_imgs.view(n_row ** 2,1,28,28))

    save_image(gen_imgs.data, "images/{}.png" .format(batches_done), nrow=n_row, normalize=True)

def sampleS_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    pz=HypersphericalUniform(Z_DIM - 1)
    z = Variable(Tensor(pz.sample(n_row ** 2).numpy()))
    gen_imgs = modelS.decode(z)
    gen_imgs=nn.Tanh()(gen_imgs.view(n_row ** 2,1,28,28))

    save_image(gen_imgs.data, "images/{}.png" .format(batches_done), nrow=n_row, normalize=True)

if __name__ =='__main__':
    for i in range(0,8):
        sampleN_image(10, 'A{}'.format(i))
        sampleS_image(10, 'B{}'.format(i))