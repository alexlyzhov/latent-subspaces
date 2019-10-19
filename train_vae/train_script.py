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

    message = ""
    
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

if __name__ == "__main__":
   
    ### Data
   
    ROOT = "../data/"
    H, W = 256, 256
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    class FaceData(Dataset):
        def __init__(self):

            self.img_name = os.listdir(ROOT)          

        def __len__(self):
            return len(self.img_name)

        def __getitem__(self, index):
            while 1:
                try:
                    img_path = os.path.join(ROOT, self.img_name[index])

                    img = Image.open(img_path)
                    img = img.resize((H, W), Image.ANTIALIAS)
                    img = np.array(img)

                    return img.transpose(2, 0, 1) / 255.
                except:
                    print("Problem with index {}.".format(index))
                    index = np.random.randint(len(self))
        
    
    datasets = {phase: FaceData() for phase in ["train"]}
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=16, shuffle=True, num_workers=0) 
               for phase in ["train"]}
    
    ### Model 
    
    model = VAE1(hid_dim=256, KOF=24, p=0.04)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=2)
    
    ### Init dirs where to save logs and weights
    
    model_dir, df_path, txt_path = init_dirs()
    
    
    #### Train
    model = model.to(device)
    df = pd.DataFrame(columns=['epoch', 'phase', 'itr', 'lr', 'KL', 'RE', 'beta', 'time'])
    
    
    ELBOs = {}
    KLs = {}
    REs = {}
    
    beta=0.1
    
    best_losses = [10000.]
    num_saved_models = 0

    for epoch in range(6):
        ELBOs[epoch] = []
        KLs[epoch] = []
        REs[epoch] = []

        for phase in ["train"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            start = time.time()
            for i, X_batch in enumerate(dataloaders[phase]):


                KL, RE, loss = model.calculate_loss(X_batch.float().to(device), average=True, beta=beta)
                ELBOs[epoch].append(loss.item())
                KLs[epoch].append(KL.item())
                REs[epoch].append(RE.item())

                if phase == "train":
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

                if i % 10 == 0:
                    print("{}ep: {}: {}itr: loss={}, KL={}, RE={}".format(epoch, phase, i, 
                                                                          loss.item(), KL.item(),
                                                                          RE.item()))

                if i % 10 == 0:
                    time_elapsed = time.time() - start
                    start = time.time()
                    
                    df.loc[df.shape[0]] = [epoch+1, phase, i, get_lr(optimizer), 
                                           np.mean(KLs[epoch]), np.mean(REs[epoch]), 0.1,
                                           '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)]
                    df.to_csv(df_path, index=False)
                    
                    if (np.mean(ELBOs[epoch])  < np.array(best_losses)).any():
                        if 10000. in best_losses:
                            best_losses.remove(10000.)

                        # Save new best model
                        print("--saving best model--")
                        best_losses.append(np.mean(ELBOs[epoch]))
                        print("Best losses:", best_losses)
                        model_suffix = "weight.epoch_{}_loss_val_{}.pth".format(epoch+1, np.mean(ELBOs[epoch]))
                        model_path = os.path.join(model_dir, model_suffix)

                        torch.save(model.state_dict(), model_path)
                        num_saved_models += 1
                        
                        # Plots to save
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
                        
                        img_suffix = "img.epoch_{}_loss_val_{}.png".format(epoch+1, np.mean(ELBOs[epoch]))
                        img_path = os.path.join(model_dir + "/imgs", img_suffix)
                        plt.savefig(img_path)
                        model.train()

                        # If there are more then 2 saved models, remove the worst one
                        if num_saved_models > 2:
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
    
        scheduler.step()
    
    
    