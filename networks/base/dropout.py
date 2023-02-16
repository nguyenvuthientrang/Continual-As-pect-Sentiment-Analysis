import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter



class GaussDropout(nn.Module):
    def __init__(self, input_size=768, p=0.5):
        """
            Variational Dropout
            :param input_size: An int of input size
            :param p: An initial variance of noise / drop rate
        """
        super(GaussDropout, self).__init__()

        self.input_size = input_size

        # Initial alpha
        alpha = math.sqrt(p/(1-p))
        self.max_alpha = 1.0    # set threshold for alpha
        log_alpha = (torch.ones(1, self.input_size) * alpha).log()        
        self.log_alpha = nn.Parameter(log_alpha)        # learn log_alpha 


    def kld(self):
        """
            Calculate KL-divergence between N(1, alpha) and log-uniform prior
            This approximated KL is calculated follow the Kingma's paper
            https://arxiv.org/abs/1506.02557
        """

        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = (2 * self.log_alpha).exp()         # self.log_alpha was clipped to ensure less equal than zero   
        negative_kl = self.log_alpha + c1*alpha + c2*alpha**2 + c3*alpha**3          
        kl = -negative_kl
        
        return kl.sum()

    def forward(self, x, noise= None):
        """
            :param x: An float tensor with shape of [batch_size, input_size]
            :param noise: noise for testing model
            :return: An float tensor with shape of [batch_size, input_size] and layer-kld estimation
        """

        kld = 0
        if self.training:
            # N(0,1)
            epsilon = torch.randn(x.size()) 
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max= math.log(self.max_alpha - 1e-6))
            alpha = torch.exp(self.log_alpha)

            # input vectors have the same alpha vector
            # each feature in each input vector has particular alpha_{i} 
            epsilon = epsilon * alpha + 1
            kld = self.kld()
            
            return x * epsilon, kld

        else:
            s = 1
            if noise != None:
                log_alpha = torch.Tensor(noise)
                epsilon = torch.randn(x.size())    
                if x.is_cuda:
                    epsilon = epsilon.cuda()
                log_alpha = torch.clamp(log_alpha, max= math.log(self.max_alpha - 1e-6))
                alpha = torch.exp(log_alpha)
                # alpha = math.exp(noise)
                s = epsilon * alpha + 1
            # No scaling 
            return x * s, kld