import math
import torch
from torch.distributions.kl import register_kl

from hyperspherical_vae.ops.ive import ive
from hyperspherical_vae.distributions.hyperspherical_uniform import HypersphericalUniform

seed=1
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda:0" if use_cuda else "cpu")

class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.] + [0] * (loc.shape[-1] - 1))).to(self.device)

        super(VonMisesFisher, self).__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)

        v = (torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape)).to(self.device).transpose(0, -1)[1:]).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w.view(-1,1), w_.view(-1,1) * v), -1)
        z = self.householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        '''
        :param shape:
        :return: 返回分布在-1,1区间的随机变量
        '''
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        #w=h(u)=log(u+(1-u)*exp(-2*s))/s+1 将分布在0,1的u映射到-1,1区间
        #2012 Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)
        self.__w = 1 + torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)#c=√(4k^2+(m-1)^2 )
        b_true = (-2 * self.scale + c) / (self.__m - 1)# b=((m-1)+2k+c)/(m-1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(torch.max(torch.tensor([0.], device=self.device),
                                self.scale - 10), torch.tensor([1.], device=self.device))
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4#a=((m-1)+2k+c)/4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)#d=4ab/(1+b)-(m-1)ln(m-1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w

    def __while_loop(self, b, a, d, shape):

        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b).to(self.device), torch.zeros_like(
            b).to(self.device), (torch.ones_like(b) == 1).to(self.device)

        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            e_ = torch.distributions.Beta((self.__m - 1) / 2, (self.__m - 1) /
                                          2).sample(shape[:-1]).reshape(shape).to(self.device)
            u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * t.log() - t + d) > torch.log(u)
            reject = 1 - accept

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e, w

    def householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        output = - self.scale * ive(self.__m / 2, self.scale) / ive((self.__m / 2) - 1, self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale) - (self.__m / 2) * math.log(2 * math.pi) - (
            self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return - vmf.entropy() + hyu.entropy().to(device)

@register_kl(VonMisesFisher, VonMisesFisher)
def _kl_vmf_uniform(vmf1, vmf2):
    return - vmf1.entropy() +vmf2.entropy()

if __name__ == '__main__':
    x=VonMisesFisher(torch.tensor([0,0,1],dtype=torch.float),torch.tensor(2,dtype=torch.float))
    s=x.sample(4)
    print(s)
    # import numpy as np
    # import matplotlib.pyplot as plt
    # x=np.arange(0,1,0.01)
    # s=2
    # f=lambda u:np.log(u+(1-u)*np.exp(-2*s))/s+1
    # y=f(x)
    # plt.plot(x,y)
    # plt.show()