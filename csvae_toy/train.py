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

from definitions import VAE, CSVAE_without_delta_net
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--tag', type=str)
args = parser.parse_args()

assert args.model in ['VAE', 'CSVAE_without_delta_net']  # replace


x_train, manifold_x_train = make_swiss_roll(n_samples=10000)
x_train = x_train.astype(np.float32)
y_train = (x_train[:, 0:1] >= 10).astype(np.float32)

x_test, manifold_x_test = make_swiss_roll(n_samples=10000)
x_test = x_test.astype(np.float32)
y_test = (x_test[:, 0:1] >= 10).astype(np.float32)

z_dim = 2
w_dim = 2
batch_size = 32

if args.model == 'VAE':
    train_set_tensor = torch.from_numpy(x_train).cuda()
    train_set = utils.TensorDataset(train_set_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set_tensor = torch.from_numpy(x_test).cuda()
    
    model = VAE(input_dim, z_dim, w_dim, beta=1).train().cuda()
elif args.model == 'CSVAE_without_delta_net':
    train_set_x_tensor = torch.from_numpy(x_train).cuda()
    train_set_y_tensor = torch.from_numpy(y_train).cuda()
    train_set = utils.TensorDataset(train_set_x_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set_x_tensor = torch.from_numpy(x_test).cuda()
    test_set_y_tensor = torch.from_numpy(y_test).cuda()

    model = CSVAE_without_delta_net(input_dim=x_train.shape[1], labels_dim=y_train.shape[1], z_dim=z_dim, w_dim=w_dim).train().cuda()
    
opt = optim.Adam(model.parameters(), lr=1e-3/2)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
n_epochs = 2300

root_dir = 'res'
png_dir = os.path.join(root_dir, args.tag, 'latent_vis')
models_dir = os.path.join(root_dir, args.tag, 'models')
os.makedirs(png_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


mse_losses = []
w_kl_losses = []
z_kl_losses = []
for epoch_i in trange(n_epochs):
    for cur_batch in train_loader:
        opt.zero_grad()
        if args.model == 'VAE':
            cur_batch = cur_batch[0]
            opt.zero_grad()
            mse_loss_val, kl_loss_val, _ = model.calculate_loss(cur_batch)
            loss_val = mse_loss_val + kl_loss_val*beta
            loss_val.backward()
            opt.step()
            mse_losses.append(mse_loss_val.item())
            z_kl_losses.append((kl_loss_val*beta).item())
        elif args.model == 'CSVAE_without_delta_net':
            cur_batch = cur_batch
            loss_val, recon_loss_val, z_kl_loss_val, w_kl_loss_val = model.calculate_loss(*cur_batch)
            loss_val.backward()
            opt.step()
            mse_losses.append(recon_loss_val.item())
            z_kl_losses.append(z_kl_loss_val.item())
            w_kl_losses.append(w_kl_loss_val.item())
    scheduler.step()
    print(f'Epoch {epoch_i}')
    print(f'Mean MSE: {np.array(mse_losses[-len(train_loader):]).mean():.4f}')
    if args.model == 'VAE':
        print(f'Mean KL: {np.array(z_kl_losses[-len(train_loader):]).mean():.4f}')
    elif args.model == 'CSVAE_without_delta_net':
        print(f'Mean w KL: {np.array(w_kl_losses[-len(train_loader):]).mean():.4f}')
        print(f'Mean z KL: {np.array(z_kl_losses[-len(train_loader):]).mean():.4f}')
    print()
    
    if args.model == 'VAE':
        _, _, zw, _, _ = model.forward(test_set_tensor)
    elif args.model == 'CSVAE_without_delta_net':
        _, _, zw, _, _, _, _, _, _ = model.forward(test_set_x_tensor, test_set_y_tensor)

    colors_test = ['red' if label else 'blue' for label in y_test]

    zw = zw.detach().cpu().numpy()
    z_comp = zw[:, :2]
    w_comp = zw[:, 2:]
    
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
    