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
import matplotlib.pyplot as plt

from definitions import VAE

xyz, manifold_x = make_swiss_roll(n_samples=10000)
xyz = xyz.astype(np.float32)
# labels = (xyz[:, 0] >= 10)
input_dim = xyz.shape[1]
z_dim = 2
w_dim = 2

batch_size = 32
beta = 1

model = VAE(input_dim, z_dim, w_dim, beta)
model = model.train().cuda()

train_set_tensor = torch.from_numpy(xyz)
train_set = utils.TensorDataset(train_set_tensor)
train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

xyz_test, manifold_x_test = make_swiss_roll(n_samples=10000)
xyz_test = xyz_test.astype(np.float32)
test_set_tensor = torch.from_numpy(xyz_test).cuda()

opt = optim.Adam(model.parameters(), lr=1e-3/2)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
n_epochs = 2300

root_dir = 'res'
# exp_name = 'vae_roll_30_epochs'
exp_name = 'vae_roll_with_csvae_schedule'
png_dir = os.path.join(root_dir, exp_name, 'latent_vis')
models_dir = os.path.join(root_dir, exp_name, 'models')
os.makedirs(png_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mse_losses = []
kl_losses = []
for epoch_i in trange(n_epochs):
    for cur_batch in train_loader:
        cur_batch = cur_batch[0].cuda()
        opt.zero_grad()
        mse_loss_val, kl_loss_val, _ = model.calculate_loss(cur_batch)
        loss_val = mse_loss_val + kl_loss_val*beta
        loss_val.backward()
        opt.step()
        mse_losses.append(mse_loss_val.item())
        kl_losses.append((kl_loss_val*beta).item())
    scheduler.step()
    print(f'Epoch {epoch_i}')
    print(f'Mean MSE: {np.array(mse_losses[-len(train_loader):]).mean():.4f}')
    print(f'Mean KL: {np.array(kl_losses[-len(train_loader):]).mean():.4f}')
    print()
    
    mu_x, logvar_x, z_hat, mu_z, logvar_z = model.forward(test_set_tensor)

    labels_test = xyz_test[:, 0] >= 10
    colors_test = ['red' if label else 'blue' for label in labels_test]

    z_hat = z_hat.detach().cpu().numpy()
    z_comp = z_hat[:, :2]
    w_comp = z_hat[:, 2:]
    
    cur_title = f'(z1, z2), epoch {epoch_i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(z_comp[:, 0], z_comp[:, 1], c=colors_test)
    plt.savefig(os.path.join(png_dir, cur_title))

    cur_title = f'(z2, w1), epoch {epoch_i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(z_comp[:, 1], w_comp[:, 0], c=colors_test)
    plt.savefig(os.path.join(png_dir, cur_title))

    cur_title = f'(w1, w2), epoch {epoch_i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(w_comp[:, 0], w_comp[:, 1], c=colors_test)
    plt.savefig(os.path.join(png_dir, cur_title))

    cur_title = f'(w2, z1), epoch {epoch_i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(w_comp[:, 1], w_comp[:, 0], c=colors_test)
    plt.savefig(os.path.join(png_dir, cur_title))
    
    plt.close('all')
    
    torch.save(model, os.path.join(models_dir, 'vae.pt'))  # may slow down training if arch is large!
    