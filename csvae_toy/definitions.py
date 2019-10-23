import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm, trange

class VAE(nn.Module):
    def __init__(self, input_dim, z_dim, w_dim, beta): 
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.beta = beta
        self.encoder = nn.Sequential(nn.Linear(input_dim, z_dim+w_dim), nn.ReLU(), nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU())
        self.mu_encoder = nn.Linear(z_dim+w_dim, z_dim+w_dim)
        self.logvar_encoder = nn.Linear(z_dim+w_dim, z_dim+w_dim)
        
        self.decoder = nn.Sequential(nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU(), nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU())
        self.mu_decoder = nn.Linear(z_dim+w_dim, input_dim)
        self.logvar_decoder = nn.Linear(z_dim+w_dim, input_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        
    def q_z(self, x):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """

        intermediate = self.encoder(x)
        mu = self.mu_encoder(intermediate)
        logvar = self.logvar_encoder(intermediate)
        
        return mu, logvar

    def p_x(self, z):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        intermediate = self.decoder(z)
        mu = self.mu_decoder(intermediate)
        logvar = self.logvar_decoder(intermediate)
        
        return mu, logvar

    def forward(self, x):
        """
        Encode the image, sample z and decode 
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        mu_z, logvar_z = self.q_z(x)
        z_hat = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.p_x(z_hat)
        return mu_x, logvar_x, z_hat, mu_z, logvar_z

    def log_p_z(self, z):
        """
        Log density of the Prior
        :param z: latent vector     (MB, hid_dim)
        :return: \sum_i log p(z_i)  (1, )
        """
        out = -0.5 * z.shape[0]*z.shape[1] * torch.log(torch.tensor(2 * np.pi)) + (- (z**2).sum(dim=1) / 2).exp_()

        return out

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    def kl(self, z, z_mean, z_logvar):
        """
        KL-divergence between p(z) and q(z|x)
        :param z:                               (MB, hid_dim)
        :param z_mean: mean of q(z|x)           (MB, hid_dim)
        :param z_logvar: log variance of q(z|x) (MB, hid_dim)
        :return: KL                             (MB, )
        """

        out = - 0.5 * (1 + z_logvar - z_mean**2 - z_logvar.exp()).mean(dim=1)
        return out

    def calculate_loss(self, x, average=True):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """
        mu_x, logvar_x, z_hat, mu_z, logvar_z = self.forward(x)

        KL = self.kl(z_hat, mu_z, logvar_z)
        RE = ((torch.sigmoid(mu_x) - x)**2).mean(dim=(1))
        
        ELBO = self.beta * KL + RE
        
        if average:
            ELBO = ELBO.mean()
            KL = KL.mean()
            RE = RE.mean()

        return KL, RE, ELBO

    def calculate_nll(self, X, samples=5000):
        """
        Estimate NLL by importance sampling
        :param X: dataset, (N, inp_dim)
        :param samples: Samples per observation
        :return: IS estimate
        """   
        prob_sum = 0.

        for i in range(samples):
            KL, RE, _ = self.calculate_loss(X)
            prob_sum += (KL + RE).exp_()
            
        return - (prob_sum / samples).sum().log_()

    def generate_x(self, N=25):
        """
        Sample, using you VAE: sample z from prior and decode it 
        :param N: number of samples
        :return: X (N, inp_size)
        """

        m = MultivariateNormal(torch.zeros(self.z_dim + self.w_dim), torch.eye(self.z_dim + self.w_dim))
        z = m.sample(sample_shape=torch.Size([N])) 
        
        X, _ = self.p_x(z.cuda())
        return X

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)