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
    
    
class CSVAE_without_delta_net(nn.Module):
    def __init__(self, input_dim, labels_dim, z_dim, w_dim):
        super(CSVAE_without_delta_net, self).__init__()
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.encoder_xy_to_w = nn.Sequential(nn.Linear(input_dim+labels_dim, w_dim), nn.ReLU(), nn.Linear(w_dim, w_dim), nn.ReLU())
        self.mu_xy_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_xy_to_w = nn.Linear(w_dim, w_dim)
        
        self.encoder_x_to_z = nn.Sequential(nn.Linear(input_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU())
        self.mu_x_to_z = nn.Linear(z_dim, z_dim)
        self.logvar_x_to_z = nn.Linear(z_dim, z_dim)
        
        self.encoder_y_to_w = nn.Sequential(nn.Linear(labels_dim, w_dim), nn.ReLU(), nn.Linear(w_dim, w_dim), nn.ReLU())
        self.mu_y_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_y_to_w = nn.Linear(w_dim, w_dim)
        
        # Add sigmoid or smth for images!
        self.decoder_zw_to_x = nn.Sequential(nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU(), nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU())
        self.mu_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)
        self.logvar_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        
    def q_zw(self, x, y):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """
        xy = torch.cat([x, y], dim=1)
        
        intermediate = self.encoder_x_to_z(x)
        z_mu = self.mu_x_to_z(intermediate)
        z_logvar = self.mu_x_to_z(intermediate)
        
        intermediate = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(intermediate)
        w_logvar_encoder = self.mu_xy_to_w(intermediate)
        
        intermediate = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(intermediate)
        w_logvar_prior = self.mu_y_to_w(intermediate)
        
        return w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar
    
    def p_x(self, z, w):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        zw = torch.cat([z, w], dim=1)
        
        intermediate = self.decoder_zw_to_x(zw)
        mu = self.mu_zw_to_x(intermediate)
        logvar = self.logvar_zw_to_x(intermediate)
        
        return mu, logvar

    def forward(self, x, y):
        """
        Encode the image, sample z and decode 
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)
        z = self.reparameterize(z_mu, z_logvar)
        zw = torch.cat([z, w_encoder], dim=1)
        
        x_mu, x_logvar = self.p_x(z, w_encoder)
        
        return x_mu, x_logvar, zw, \
               w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def calculate_loss(self, x, y, average=True):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """
        x_mu, x_logvar, zw, \
            w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        
        z_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]).cuda(), torch.eye(self.z_dim * z_mu.size()[0]).cuda())
        
        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp()))
        
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)

        recon = ((x_mu - x)**2).mean(dim=(1))
        # alternatively use predicted logvar too to evaluate density of input
        
        ELBO = 20 * recon + 0.2 * z_kl + 1 * w_kl
        
        if average:
            ELBO = ELBO.mean()
            recon = recon.mean()
            z_kl = z_kl.mean()
            w_kl = w_kl.mean()

        return ELBO, recon, z_kl, w_kl

#     def reconstruct_x(self, x, y):
#         x_mean, _, _, _, _ = self.forward(x, y)
#         return x_mean

#     def calculate_nll(self, X, samples=5000):
#         """
#         Estimate NLL by importance sampling
#         :param X: dataset, (N, inp_dim)
#         :param samples: Samples per observation
#         :return: IS estimate
#         """   
#         prob_sum = 0.

#         for i in range(samples):
#             KL, RE, _ = self.calculate_loss(X)
#             prob_sum += (KL + RE).exp_()
            
#         return - (prob_sum / samples).sum().log_()

#     def generate_x(self, N=25):
#         """
#         Sample, using you VAE: sample z from prior and decode it 
#         :param N: number of samples
#         :return: X (N, inp_size)
#         """

#         m = MultivariateNormal(torch.zeros(self.z_dim + self.w_dim), torch.eye(self.z_dim + self.w_dim))
#         z = m.sample(sample_shape=torch.Size([N])) 
        
#         X, _ = self.p_x(z.cuda())
#         return X

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)
    
    
class CSVAE(nn.Module):
    def __init__(self, input_dim, labels_dim, z_dim, w_dim):
        super(CSVAE, self).__init__()
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.encoder_xy_to_w = nn.Sequential(nn.Linear(input_dim+labels_dim, w_dim), nn.ReLU(), nn.Linear(w_dim, w_dim), nn.ReLU())
        self.mu_xy_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_xy_to_w = nn.Linear(w_dim, w_dim)
        
        self.encoder_x_to_z = nn.Sequential(nn.Linear(input_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU())
        self.mu_x_to_z = nn.Linear(z_dim, z_dim)
        self.logvar_x_to_z = nn.Linear(z_dim, z_dim)
        
        self.encoder_y_to_w = nn.Sequential(nn.Linear(labels_dim, w_dim), nn.ReLU(), nn.Linear(w_dim, w_dim), nn.ReLU())
        self.mu_y_to_w = nn.Linear(w_dim, w_dim)
        self.logvar_y_to_w = nn.Linear(w_dim, w_dim)
        
        # Add sigmoid or smth for images!
        self.decoder_zw_to_x = nn.Sequential(nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU(), nn.Linear(z_dim+w_dim, z_dim+w_dim), nn.ReLU())
        self.mu_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)
        self.logvar_zw_to_x = nn.Linear(z_dim+w_dim, input_dim)
        
        self.decoder_z_to_y = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU(),
                                            nn.Linear(z_dim, labels_dim), nn.Sigmoid())

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        
    def q_zw(self, x, y):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """
        xy = torch.cat([x, y], dim=1)
        
        intermediate = self.encoder_x_to_z(x)
        z_mu = self.mu_x_to_z(intermediate)
        z_logvar = self.mu_x_to_z(intermediate)
        
        intermediate = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(intermediate)
        w_logvar_encoder = self.mu_xy_to_w(intermediate)
        
        intermediate = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(intermediate)
        w_logvar_prior = self.mu_y_to_w(intermediate)
        
        return w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar
    
    def p_x(self, z, w):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        zw = torch.cat([z, w], dim=1)
        
        intermediate = self.decoder_zw_to_x(zw)
        mu = self.mu_zw_to_x(intermediate)
        logvar = self.logvar_zw_to_x(intermediate)
        
        return mu, logvar

    def forward(self, x, y):
        """
        Encode the image, sample z and decode 
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)
        z = self.reparameterize(z_mu, z_logvar)
        zw = torch.cat([z, w_encoder], dim=1)
        
        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)
        
        return x_mu, x_logvar, zw, y_pred, \
               w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def calculate_loss(self, x, y):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """
        x_mu, x_logvar, zw, y_pred, \
            w_mu_encoder, w_logvar_encoder, w_mu_prior, \
            w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        
        x_recon = nn.MSELoss()(x_mu, x)
        
        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp()))
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)
        
        z_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]).cuda(), torch.eye(self.z_dim * z_mu.size()[0]).cuda())
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)
        
        y_pred_negentropy = (y_pred.log() * y_pred + (1-y_pred).log() * (1-y_pred)).mean()

        y_recon = nn.BCELoss()(y_pred, y)
        # alternatively use predicted logvar too to evaluate density of input
        
        ELBO = 20 * x_recon + 0.2 * z_kl + 1 * w_kl + 10 * y_pred_negentropy + 1 * y_recon
        
        return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon

#     def reconstruct_x(self, x, y):
#         x_mean, _, _, _, _ = self.forward(x, y)
#         return x_mean

#     def calculate_nll(self, X, samples=5000):
#         """
#         Estimate NLL by importance sampling
#         :param X: dataset, (N, inp_dim)
#         :param samples: Samples per observation
#         :return: IS estimate
#         """   
#         prob_sum = 0.

#         for i in range(samples):
#             KL, RE, _ = self.calculate_loss(X)
#             prob_sum += (KL + RE).exp_()
            
#         return - (prob_sum / samples).sum().log_()

#     def generate_x(self, N=25):
#         """
#         Sample, using you VAE: sample z from prior and decode it 
#         :param N: number of samples
#         :return: X (N, inp_size)
#         """

#         m = MultivariateNormal(torch.zeros(self.z_dim + self.w_dim), torch.eye(self.z_dim + self.w_dim))
#         z = m.sample(sample_shape=torch.Size([N])) 
        
#         X, _ = self.p_x(z.cuda())
#         return X

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)