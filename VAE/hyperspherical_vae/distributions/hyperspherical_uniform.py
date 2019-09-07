
import math
import torch


class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        '''
        从(0,1)标准正态分布中采样得到(shape,self._dim+1)的数据，并将每一行变成单位向量输出
        :param shape:
        :return:
        '''
        output = torch.distributions.Normal(0, 1).sample(
            (shape if isinstance(shape, torch.Size) else torch.Size([shape])) + torch.Size([self._dim + 1])).to(self.device)

        return output / output.norm(dim=-1, keepdim=True)#将随机采样得到的向量归一化就变成球面向量了

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return - torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        '''
        :return: 单位半径，(self._dim + 1)维超球体的表面积的自然对数值
        '''
        #torch.gamma():  the log of the gamma function
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
            torch.Tensor([(self._dim + 1) / 2], device=self.device))

