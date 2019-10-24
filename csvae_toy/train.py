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
import dill
import matplotlib.pyplot as plt

from definitions import VAE, CSVAE_without_delta_net, CSVAE
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['VAE', 'CSVAE_without_delta_net', 'CSVAE'])
parser.add_argument('--tag', type=str)
args = parser.parse_args()


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
    
    beta = 1
    model = VAE(x_train.shape[1], z_dim, w_dim, beta=beta).train().cuda()
elif args.model == 'CSVAE_without_delta_net':
    train_set_x_tensor = torch.from_numpy(x_train).cuda()
    train_set_y_tensor = torch.from_numpy(y_train).cuda()
    train_set = utils.TensorDataset(train_set_x_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set_x_tensor = torch.from_numpy(x_test).cuda()
    test_set_y_tensor = torch.from_numpy(y_test).cuda()

    model = CSVAE_without_delta_net(input_dim=x_train.shape[1], labels_dim=y_train.shape[1], z_dim=z_dim, w_dim=w_dim).train().cuda()
elif args.model == 'CSVAE':
    train_set_x_tensor = torch.from_numpy(x_train).cuda()
    train_set_y_tensor = torch.from_numpy(y_train).cuda()
    train_set = utils.TensorDataset(train_set_x_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set_x_tensor = torch.from_numpy(x_test).cuda()
    test_set_y_tensor = torch.from_numpy(y_test).cuda()
    test_set = utils.TensorDataset(test_set_x_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = CSVAE(input_dim=x_train.shape[1], labels_dim=y_train.shape[1], z_dim=z_dim, w_dim=w_dim).train().cuda()
    
n_epochs = 2300
if args.model != 'CSVAE':
    opt = optim.Adam(model.parameters(), lr=1e-3/2)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
else:
    params_without_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' not in name]
    params_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' in name]

    opt_without_delta = optim.Adam(params_without_delta, lr=(1e-3)/2)
    scheduler_without_delta = optim.lr_scheduler.MultiStepLR(opt_without_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
    opt_delta = optim.Adam(params_delta, lr=(1e-3)/2)
    scheduler_delta = optim.lr_scheduler.MultiStepLR(opt_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))

root_dir = 'res'
vis_dir = os.path.join(root_dir, args.tag, 'latent_vis')
models_dir = os.path.join(root_dir, args.tag, 'models')
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


train_x_recon_losses = []
train_w_kl_losses = []
train_z_kl_losses = []
train_y_negentropy_losses = []
train_y_recon_losses = []

test_x_recon_losses = []
test_w_kl_losses = []
test_z_kl_losses = []
test_y_negentropy_losses = []
test_y_recon_losses = []

test_every = 100

for epoch_i in trange(n_epochs):
    for cur_batch in train_loader:
        if args.model == 'VAE':
            opt.zero_grad()
            cur_batch = cur_batch[0]
            opt.zero_grad()
            x_recon_loss_val, kl_loss_val, _ = model.calculate_loss(cur_batch)
            loss_val = x_recon_loss_val + kl_loss_val*beta
            loss_val.backward()
            opt.step()
            train_x_recon_losses.append(x_recon_loss_val.item())
            train_z_kl_losses.append((kl_loss_val*beta).item())
        elif args.model == 'CSVAE_without_delta_net':
            opt.zero_grad()
            loss_val, x_recon_loss_val, z_kl_loss_val, w_kl_loss_val = model.calculate_loss(*cur_batch)
            loss_val.backward()
            opt.step()
            train_x_recon_losses.append(x_recon_loss_val.item())
            train_z_kl_losses.append(z_kl_loss_val.item())
            train_w_kl_losses.append(w_kl_loss_val.item())
        elif args.model == 'CSVAE':
            loss_val, x_recon_loss_val, w_kl_loss_val, z_kl_loss_val, y_negentropy_loss_val, y_recon_loss_val = model.calculate_loss(*cur_batch)
            
            # optimization could be done more precisely but less efficiently by only updating delta or other params on a batch
        
            opt_delta.zero_grad()
            y_recon_loss_val.backward(retain_graph=True)
            opt_delta.step()

            opt_without_delta.zero_grad()
            loss_val.backward()
            opt_without_delta.step()
        
            train_x_recon_losses.append(x_recon_loss_val.item())
            train_w_kl_losses.append(w_kl_loss_val.item())
            train_z_kl_losses.append(z_kl_loss_val.item())
            train_y_negentropy_losses.append(y_negentropy_loss_val.item())
            train_y_recon_losses.append(y_recon_loss_val.item())
    if args.model != 'CSVAE':
        scheduler.step()
    else:
        scheduler_without_delta.step()
        scheduler_delta.step()
    print(f'Epoch {epoch_i}')
    print('Train')
    print(f'MSE(x): {np.array(train_x_recon_losses[-len(train_loader):]).mean():.4f}')
    if args.model == 'VAE':
        print(f'KL:: {np.array(train_z_kl_losses[-len(train_loader):]).mean():.4f}')
    elif args.model == 'CSVAE_without_delta_net':
        print(f'KL(w): {np.array(train_w_kl_losses[-len(train_loader):]).mean():.4f}')
        print(f'KL(z): {np.array(train_z_kl_losses[-len(train_loader):]).mean():.4f}')
    elif args.model == 'CSVAE':
        print(f'KL(w): {np.array(train_w_kl_losses[-len(train_loader):]).mean():.4f}')
        print(f'KL(z): {np.array(train_z_kl_losses[-len(train_loader):]).mean():.4f}')
        print(f'-H(y): {np.array(train_y_negentropy_losses[-len(train_loader):]).mean():.4f}')
        print(f'BCE(y): {np.array(train_z_kl_losses[-len(train_loader):]).mean():.4f}')
    print()

    torch.save(model.state_dict(), os.path.join(models_dir, 'model_{epoch_i}.pt'))
    
    if epoch_i % test_every == 0:
        if args.model == 'CSVAE':
            for cur_batch in test_loader:
                loss_val, x_recon_loss_val, w_kl_loss_val, z_kl_loss_val, y_negentropy_loss_val, y_recon_loss_val = model.calculate_loss(*cur_batch)

                test_x_recon_losses.append(x_recon_loss_val.item())
                test_w_kl_losses.append(w_kl_loss_val.item())
                test_z_kl_losses.append(z_kl_loss_val.item())
                test_y_negentropy_losses.append(y_negentropy_loss_val.item())
                test_y_recon_losses.append(y_recon_loss_val.item())

            print('Test')
            print(f'MSE(x): {np.array(test_x_recon_losses[-len(train_loader):]).mean():.4f}')
            print(f'KL(w): {np.array(test_w_kl_losses[-len(train_loader):]).mean():.4f}')
            print(f'KL(z): {np.array(test_z_kl_losses[-len(train_loader):]).mean():.4f}')
            print(f'-H(y): {np.array(test_y_negentropy_losses[-len(train_loader):]).mean():.4f}')
            print(f'BCE(y): {np.array(test_z_kl_losses[-len(train_loader):]).mean():.4f}')

        if args.model == 'VAE':
            _, _, zw, _, _ = model.forward(test_set_tensor)
        elif args.model == 'CSVAE_without_delta_net':
            _, _, zw, _, _, _, _, _, _ = model.forward(test_set_x_tensor, test_set_y_tensor)
        elif args.model == 'CSVAE':
            _, _, zw, _, _, _, _, _, _, _ = model.forward(test_set_x_tensor, test_set_y_tensor)

        colors_test = ['red' if label else 'blue' for label in y_test]

        zw = zw.detach().cpu().numpy()
        z_comp = zw[:, :2]
        w_comp = zw[:, 2:]

        scatter_size = 12

        cur_title = f'(z1, z2), epoch {epoch_i}'
        plt.figure(figsize=(5, 5,))
        plt.title(cur_title)
        plt.scatter(z_comp[:, 0], z_comp[:, 1], c=colors_test, s=scatter_size)
        plt.savefig(os.path.join(vis_dir, cur_title))

        cur_title = f'(z2, w1), epoch {epoch_i}'
        plt.figure(figsize=(5, 5,))
        plt.title(cur_title)
        plt.scatter(z_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
        plt.savefig(os.path.join(vis_dir, cur_title))

        cur_title = f'(w1, w2), epoch {epoch_i}'
        plt.figure(figsize=(5, 5,))
        plt.title(cur_title)
        plt.scatter(w_comp[:, 0], w_comp[:, 1], c=colors_test, s=scatter_size)
        plt.savefig(os.path.join(vis_dir, cur_title))

        cur_title = f'(w2, z1), epoch {epoch_i}'
        plt.figure(figsize=(5, 5,))
        plt.title(cur_title)
        plt.scatter(w_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
        plt.savefig(os.path.join(vis_dir, cur_title))

        plt.close('all')
    