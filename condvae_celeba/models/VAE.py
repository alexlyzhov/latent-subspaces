import torch
from torch import nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)
    
class Conv_block(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.2, p=0.04, transpose=False):
        super(Conv_block, self).__init__()
        
        self.transpose = transpose
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            
        self.activation = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout2d(p)
        self.batch_norm = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if not self.transpose:
            x = self.dropout(x)
        x = self.batch_norm(x)

        return x

class VAE1(nn.Module):
    def __init__(self, hid_dim, KOF, p):
        super(VAE1, self).__init__()

        self.hid_dim = hid_dim
        self.encoder = nn.Sequential()
        self.encoder.add_module("block01", Conv_block(KOF, 3, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(KOF*2, KOF, KOF*2, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(KOF*4, KOF*2, KOF*4, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(KOF*8, KOF*4, KOF*8, 4, 2, 1, p=p))
        self.encoder.add_module("block05", Conv_block(KOF*16, KOF*8, KOF*16, 4, 2, 1, p=p))
        self.encoder.add_module("block06", Conv_block(KOF*16, KOF*16, KOF*16, 4, 2, 1, p=p))
#         self.encoder.add_module("block07", Conv_block(KOF*32, KOF*16, KOF*32, 4, 2, 1, p=p))
#         self.encoder.add_module("block08", Conv_block(KOF*32, KOF*32, KOF*32, 4, 2, 1, p=p))

        self.encoder.add_module("flatten", Flatten()) 
        
        self.fc1 = nn.Sequential(nn.Linear(KOF*8*2, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),nn.Dropout(p=p), nn.Linear(256, hid_dim))
        
        self.fc2 = nn.Sequential(nn.Linear(KOF*8*2, 256), nn.LeakyReLU(0.2),nn.Dropout(p=p), nn.Linear(256, hid_dim))
        

        
        self.decoder = nn.Sequential()
        self.decoder.add_module("block00", nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.LeakyReLU(0.2)))
        self.decoder.add_module("reshape", Reshape((-1, hid_dim,1,1)))
        
        self.decoder.add_module("block01", Conv_block(KOF*4, hid_dim, KOF*4, 4, 1, 0, p=p, transpose=True))
        self.decoder.add_module("block02", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
        self.decoder.add_module("block03", Conv_block(KOF*2, KOF*4, KOF*2, 3, 1, 1, p=p, transpose=True))
        self.decoder.add_module("block04", Conv_block(KOF*2, KOF*2, KOF*2, 4, 2, 1, p=p, transpose=True))
#         self.decoder.add_module("block05", Conv_block(KOF*4, KOF*4, KOF*4, 4, 2, 1, p=p, transpose=True))
#         self.decoder.add_module("block06", Conv_block(KOF*2, KOF*4, KOF*2, 4, 2, 1, p=p, transpose=True))
        self.decoder.add_module("block05", Conv_block(KOF, KOF*2, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder.add_module("block06", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder.add_module("block07", nn.Sequential(
                    nn.ConvTranspose2d(KOF, 3, 3, 1, 1)))

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

        x = self.encoder(x)
        
        mu = self.fc1(x)
        logvar = self.fc2(x)
        
        return mu, logvar

    def p_x(self, z):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        
        mu = self.decoder(z)
        logvar = self.decoder(z)
        
        return mu, logvar

    def forward(self, x):
        """
        Encoder the image, sample z and decode 
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

    def calculate_loss(self, x, beta=1., average=False):
        """
        Given the input batch, compute the negative ELBO 
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: RE + beta * KL  (MB, ) or (1, )
        """
        mu_x, logvar_x, z_hat, mu_z, logvar_z = self.forward(x)

        KL = self.kl(z_hat, mu_z, logvar_z)

        RE = ((torch.sigmoid(mu_x) - x)**2).mean(dim=(1,2,3))
        
        ELBO = beta * KL + RE
        
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

    def generate_x(self, N=25, device=torch.device("cpu")):
        """
        Sample, using you VAE: sample z from prior and decode it 
        :param N: number of samples
        :return: X (N, inp_size)
        """

        m = MultivariateNormal(torch.zeros(self.hid_dim), torch.eye(self.hid_dim))
        z = m.sample(sample_shape=torch.Size([N])) 
        
        X, _ = self.p_x(z.to(device))
        return X

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)