import torch
from torch import nn
import numpy as np
import os
import sys
import torch.nn.functional as F
from torch.autograd import Variable

from torch.distributions.multivariate_normal import MultivariateNormal

# Path initialization
CUR_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, CUR_DIR)
from VAE import VAE1

class CondVAE(nn.Module):
    def __init__(self, hid_dim, KOF, p, N_ATTRS, path=None):
        super(CondVAE, self).__init__()
        
        self.VAE = VAE1(hid_dim, KOF, p)
        if path:
            self.VAE.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
          
        self.attrs_encoder = AttributeEncoder(hid_dim, N_ATTRS)
        self.attrs_decoder = AttributeDecoder(hid_dim, N_ATTRS)
        
        self.experts = ProductOfExperts()
        
        self.hid_dim = hid_dim
    
    def forward(self, image=None, attrs=None, device=torch.device('cpu')):
        mu_z, logvar_z  = self.infer(image, attrs, device)

        z_hat = self.reparameterize(mu_z, logvar_z)

        mu_x, logvar_x = self.VAE.p_x(z_hat)
        attrs_recon = self.attrs_decoder(z_hat)
        return mu_x, attrs_recon, mu_z, logvar_z  

    def infer(self, image=None, attrs=None, device=torch.device("cpu")): 
        
        batch_size = image.size(0) if image is not None else attrs.size(0)
        
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.hid_dim), device=device)
        
        if image is not None:
            image_mu, image_logvar = self.VAE.q_z(image)
            mu     = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if attrs is not None:
            attrs_mu, attrs_logvar = self.attrs_encoder(attrs)
            mu     = torch.cat((mu, attrs_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attrs_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

class AttributeEncoder(nn.Module):
    """Parametrizes q(z|y). 
    We use a single inference network that encodes 
    all 18 features.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, hid_dim, N_ATTRS):
        super(AttributeEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_ATTRS, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish())
        self.fc1 = nn.Linear(512, hid_dim)
        self.fc2 = nn.Linear(512, hid_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar


class AttributeDecoder(nn.Module):
    """Parametrizes p(y|z).
    We use a single generative network that decodes 
    all 18 features.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, hid_dim, N_ATTRS):
        super(AttributeDecoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hid_dim, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, N_ATTRS))

    def forward(self, z):
        z = self.net(z)
        # not a one-hotted prediction: this returns a value
        # for every single index
        return z  # NOTE: no sigmoid here. See train.py
    
    
class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, device=torch.device("cpu")):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    mu, logvar = mu.to(device), logvar.to(device)
    return mu, logvar

        