import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import sys
import time
import cv2
from PIL import Image
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Dataset


# Path initialization
CUR_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(CUR_DIR, "vae_architectures"))
from VAE import VAE1


def init_dirs():
    r"""The function for initialization of name of the dirictory where 
    to save model weights, logs, file with message (where the training 
    ditails can be listed) and plots with results of model performance.
    """

    message = "Training on 64x64 images. Only MSE loss. \nPretrained=weights/VAE1_Liza_21_10_2019_10_57/weight.epoch_30_itr_159_loss_val_0.008877071030803683.pth \nbeta=0.01 \nlr=1e-4  \nbatch_size=128"
    
    username = "Liza" 
    modelname = "VAE1"
    model_full_name = "{}_{}_{}".format(modelname, username, datetime.now().strftime('%d_%m_%Y_%H_%M'))
    model_dir = "weights/" + model_full_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(model_dir + "/imgs")
        
    df_path = model_dir + "/df_logs.csv"
    txt_path = model_dir + "/details.txt"
    
    if message != "":
        f = open(txt_path,"w+")
        f.write(message)
        f.close() 
        
    return model_dir, df_path, txt_path 


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_plot(model, X_batch):
    r"""Plots and saves the results of model performance.
    """

    model.eval()
    X_rec = model.reconstruct_x(X_batch.float().to(device))
    X_gen = model.generate_x(N=1, device=device)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].imshow(X_batch[0].cpu().numpy().transpose(1, 2, 0))
    ax[0].set_title("Real", fontsize=14)
    ax[1].imshow(torch.sigmoid(X_rec[0].cpu()).detach().numpy().transpose(1, 2, 0))
    ax[1].set_title("Rec", fontsize=14)
    ax[2].imshow(torch.sigmoid(X_gen[0].cpu()).detach().numpy().transpose(1, 2, 0))
    ax[2].set_title("Gen", fontsize=14)

    img_suffix = "img.phase_{}_epoch_{}_itr_{}.png".format(phase, epoch+1, i+1)
    img_path = os.path.join(model_dir + "/imgs/" + phase, img_suffix)
    
    try:
        plt.savefig(img_path)
    except:
        os.makedirs(model_dir + "/imgs/" + phase)
        plt.savefig(img_path)
    
    
def saving(model, ELBOs, best_losses, epoch, i, X_batch, num_saved_models, max_num_of_models_to_save=3):
    
    print("--saving best model--")
    best_losses.append(np.mean(ELBOs[epoch]))
    print("Best losses:", best_losses)
    model_suffix = "weight.epoch_{}_itr_{}_loss_val_{}.pth".format(epoch+1, i+1, np.mean(ELBOs[epoch]))
    model_path = os.path.join(model_dir, model_suffix)

    torch.save(model.state_dict(), model_path)
    num_saved_models += 1

    # Plots to save
    save_plot(model, X_batch)

    # If there are more then max_num_of_models_to_save saved models, remove the worst one
    if num_saved_models > max_num_of_models_to_save:
        max_loss = 0.0
        for file_name in os.listdir(model_dir):
            if "loss" in file_name:
                loss = file_name.split("_")[-1][:-4]
                loss = float(loss)
                if loss > max_loss:
                    max_loss = loss
                    model_with_max_loss = os.path.join(model_dir, file_name)
        print("Loss to move", max_loss)          
        os.remove(model_with_max_loss)
        best_losses.remove(max_loss)
        num_saved_models -= 1
        
    return num_saved_models



if __name__ == "__main__":
   
    ### Data
   
    ROOT = "../data/img_align_celeba"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    class FaceData(Dataset):
        def __init__(self, img_names):

            self.img_name = img_names         

        def __len__(self):
            return len(self.img_name)

        def __getitem__(self, index):

            img_path = os.path.join(ROOT, self.img_name[index])

            img = Image.open(img_path)
            img = img.resize((256, 256), Image.ANTIALIAS)
            img = img.crop((32, 32, 224, 224))
            img = img.resize((64, 64), Image.ANTIALIAS)
            img = np.array(img)

            return img.transpose(2, 0, 1) / 255.
      
    # Train and val imgs
    df_split = pd.read_csv("../data/dataframes/train_val_test_split.csv")
    imgs_train = df_split[df_split.phase == "train"].img_names.values[0]
    imgs_train = [name for name in imgs_train.split("'") if ".jpg" in name]
    imgs_val = df_split[df_split.phase == "val"].img_names.values[0]
    imgs_val = [name for name in imgs_val.split("'") if ".jpg" in name]

    img_names = {"train": imgs_train, "val": imgs_val}
    datasets = {phase: FaceData(img_names[phase]) for phase in ["train", "val"]}
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=128, shuffle=True, num_workers=2) 
               for phase in ["train", "val"]}
    
    ### Model 
    model = VAE1(hid_dim=128, KOF=32, p=0.04)
    model.load_state_dict(torch.load("weights/VAE1_Liza_21_10_2019_10_57/weight.epoch_30_itr_159_loss_val_0.008877071030803683.pth", map_location=device))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    
    ### Train
    n_epoch = 100
    model = model.to(device)
    df = pd.DataFrame(columns=['epoch', 'phase', 'itr', 'lr', 'KL', 'RE', 'beta', 'time'])


    ELBOs, KLs, REs = {}, {}, {}
    beta=0.001

    best_losses = [10000.]
    num_saved_models = 0

    ii = 0
    for epoch in range(n_epoch):
        ELBOs[epoch] = []
        KLs[epoch] = []
        REs[epoch] = []

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            start = time.time()
            for i, X_batch in enumerate(dataloaders[phase]):
    #             print(i)

                KL, RE, loss = model.calculate_loss(X_batch.float().to(device), average=True, beta=beta)
                ELBOs[epoch].append(loss.item())
                KLs[epoch].append(KL.item())
                REs[epoch].append(RE.item())

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if ii == 0:
                    # Init dirs where to save logs and weights
                    model_dir, df_path, txt_path = init_dirs()

                ii += 1

                if i % 25 == 24:
                    time_elapsed = time.time() - start
                    start = time.time()

                    print("{}ep: {}: {}itr: loss={}, KL={}, RE={}, time={:.0f}m {:.0f}s".format(epoch+1, phase, i+1, 
                                                                              loss.item(), KL.item(),
                                                                              RE.item(),
                                                                              time_elapsed//60, time_elapsed % 60))

                    df.loc[df.shape[0]] = [epoch+1, phase, i+1, get_lr(optimizer), 
                                           np.mean(KLs[epoch]), np.mean(REs[epoch]), 0.1,
                                           '{:.0f}m {:.0f}s'.format(time_elapsed // 60, int(time_elapsed % 60))]
                    df.to_csv(df_path, index=False)
                    
                if i % 500 == 499:
                    save_plot(model, X_batch)
                    if phase == "train":
                        model.train()

            # At the end of epoch:
            if (np.mean(ELBOs[epoch])  < np.array(best_losses)).any() and phase == "val":
                if 10000. in best_losses:
                    best_losses.remove(10000.)

                # Save new best model
                num_saved_models = saving(model, ELBOs, best_losses, 
                                          epoch, i, X_batch, num_saved_models, 
                                          max_num_of_models_to_save=6)
              
            # Print results 
            time_elapsed = time.time() - start      
            print("FINALLY: {}ep: {}: {}itr: loss={}, KL={}, RE={}, time={:.0f}m {:.0f}s".format(epoch+1, phase, i+1, 
                                                                              loss.item(), KL.item(),
                                                                              RE.item(),
                                                                              time_elapsed//60, time_elapsed % 60))
            
            # Save logs
            df.loc[df.shape[0]] = [epoch+1, phase, i+1, get_lr(optimizer), 
                                           np.mean(KLs[epoch]), np.mean(REs[epoch]), 0.1,
                                           '{:.0f}m {:.0f}s'.format(time_elapsed // 60, int(time_elapsed % 60))]
            df.to_csv(df_path, index=False)
            
        scheduler.step()

