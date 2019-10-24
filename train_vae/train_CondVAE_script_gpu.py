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
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Dataset

# Path initialization
CUR_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(CUR_DIR, "vae_architectures"))
from VAE import VAE1
from CondVAE import CondVAE

VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}
# go from label index to interpretable index
ATTR_TO_IX_DICT  = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18, 
                    'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38, 
                    'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21, 
                    'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26, 
                    'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32, 
                    'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22, 
                    'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3, 
                    'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23, 
                    'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
# we only keep 18 of the more visually distinctive features
# See [1] Perarnau, Guim, et al. "Invertible conditional gans for 
#         image editing." arXiv preprint arXiv:1611.06355 (2016).
ATTR_IX_TO_KEEP  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT  = {v:k for k, v in ATTR_TO_IX_DICT.items()}
N_ATTRS          = len(ATTR_IX_TO_KEEP)
ATTR_TO_PLOT     = ['Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Wavy_Hair']\


def init_dirs():
    r"""The function for initialization of name of the dirictory where 
    to save model weights, logs, file with message (where the training 
    ditails can be listed) and plots with results of model performance.
    """

    message = "Ditails: \ncoeffs for loss: lambda_image=1., lambda_attrs=0.01, annealing_factor=0.001 \nlr=1e-4"
    
    username = "Liza" 
    modelname = "CondVAE"
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

def save_plot(model, X_batch, attrs):
    r"""Plots and saves the results of model performance.
    """

    model.eval()
    recon_image_1, recon_attrs_1, mu_1, logvar_1 = model(X_batch, attrs, device=device)
    recon_image_2, recon_attrs_2, mu_2, logvar_2 = model(X_batch, device=device)
    recon_image_3, recon_attrs_3, mu_3, logvar_3 = model(attrs=attrs, device=device)
        
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(X_batch[0].cpu().numpy().transpose(1, 2, 0))
    ax[0].set_title("Real", fontsize=14)
    
    ax[1].imshow(torch.sigmoid(recon_image_1[0]).detach().cpu().numpy().transpose(1, 2, 0))
    ax[1].set_title("Rec joint", fontsize=14)
    ax[2].imshow(torch.sigmoid(recon_image_2[0]).detach().cpu().numpy().transpose(1, 2, 0))
    ax[2].set_title("Rec", fontsize=14)
    ax[3].imshow(torch.sigmoid(recon_image_3[0]).detach().cpu().numpy().transpose(1, 2, 0))
    ax[3].set_title("Gen", fontsize=14)

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

def get_attr(img_name):
    df_attr_img = df_attr[df_attr.img_name == img_name]
    attr = df_attr_img.values[0, 1:]
    attr   = np.array(attr).astype(int)
    attr[attr < 0] = 0
    attr = torch.from_numpy(attr).float()
    return attr[ATTR_IX_TO_KEEP]

def tensor_to_attributes(tensor):
    """Use this for the <image_transform>.
    @param tensor: PyTorch Tensor
                   D dimensional tensor
    @return attributes: list of strings
    """
    attrs  = []
    n      = tensor.size(0)
    tensor = torch.round(tensor)
    
    for i in range(n):
        if tensor[i] > 0.5:
            attr = IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[i]]
            attrs.append(attr)
    return attrs


def elbo_loss(recon_image, image, recon_attrs, attrs, mu, logvar,
              lambda_image=1.0, lambda_attrs=1.0, annealing_factor=1):
    
    image_mse, attrs_bce = 0, 0  # default params
    
#     if recon_image is not None and image is not None:
#         image_bce = torch.sum(binary_cross_entropy_with_logits(
#             recon_image.view(-1, 3 * 64 * 64), 
#             image.view(-1, 3 * 64 * 64)), dim=1)

    if recon_image is not None and image is not None:
        image_mse = ((torch.sigmoid(recon_image) - image)**2).mean(dim=(1,2,3))

    if recon_attrs is not None and attrs is not None:
        for i in range(N_ATTRS):
            attr_bce = binary_cross_entropy_with_logits(
                recon_attrs[:, i], attrs[:, i])
            attrs_bce += attr_bce

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_mse + lambda_attrs * attrs_bce 
                      + annealing_factor * KLD)
#     print(image_bce, attrs_bce, KLD)
    return ELBO


def binary_cross_entropy_with_logits(input, target):
    
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target 
            + torch.log(1 + torch.exp(-torch.abs(input))))


if __name__ == "__main__":
   
    ### Data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

#     ROOT = "../../img_align_celeba"
    ROOT = '/home/aglyzhov/celeba/img_align_celeba'

    class FaceData_with_Attributes(Dataset):
        def __init__(self, img_names, image_transform=None, attr_transform=None):

            self.img_names = img_names

            self.size = int(len(img_names))

            self.image_transform = image_transform
            self.attr_transform  = attr_transform

        def __len__(self):
            return len(self.img_name)

        def __getitem__(self, index):

            # attr
            attr = get_attr(self.img_names[index])

            if self.attr_transform is not None:
                attr = self.attr_transform(attr)

            # img 
            img_path = os.path.join(ROOT, self.img_names[index])

            img = Image.open(img_path)
            img = img.resize((256, 256), Image.ANTIALIAS)
            img = img.crop((32, 32, 224, 224))
            img = img.resize((64, 64), Image.ANTIALIAS)
            if self.image_transform is not None:
                img  = self.image_transform(img)

            img = np.array(img)

    #         return img.transpose(2, 0, 1) / 255., attr
            return img, attr

        def __len__(self):
            return self.size
        
    # Attr  
    df_attr = pd.read_csv("../data/df_attr.csv")
    
    # Train and val imgs
#     df_split = pd.read_csv("../data/dataframes/train_val_test_split.csv")
#     imgs_train = df_split[df_split.phase == "train"].img_names.values[0]
#     imgs_train = [name for name in imgs_train.split("'") if ".jpg" in name]
#     imgs_val = df_split[df_split.phase == "val"].img_names.values[0]
#     imgs_val = [name for name in imgs_val.split("'") if ".jpg" in name]
    
    preprocess_data = transforms.Compose([transforms.Resize(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])

    img_names = {"train": df_attr.img_name.values[:55000], "val": df_attr.img_name.values[55000:]}
    datasets = {phase: FaceData_with_Attributes(img_names[phase], image_transform=preprocess_data) 
                for phase in ["train", "val"]}
#     datasets = {phase: FaceData_with_Attributes(df_attr.img_name.values[:65000], image_transform=preprocess_data) 
#                 for phase in ["train", "val"]}
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=128, shuffle=True, num_workers=2) 
               for phase in ["train", "val"]}
    
    
    ### Model 
    model = CondVAE(128, 32, 0.04, N_ATTRS)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=100)
    
    ### Train
    n_epoch = 100
    model = model.to(device)
    df = pd.DataFrame(columns=['epoch', 'phase', 'itr', 'lr', 'img', 'attr', 'joint', 'loss',  'time'])


    ELBOs, JOINT_loss, IMG_loss, ATTR_loss = {}, {}, {}, {}
    beta=0.001

    best_losses = [10000.]
    num_saved_models = 0

    ii = 0
    for epoch in range(n_epoch):
        ELBOs[epoch] = []
        JOINT_loss[epoch] = []
        IMG_loss[epoch] = []
        ATTR_loss[epoch] = []

        for phase in ["train"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            start = time.time()
            for i, (image, attrs) in enumerate(dataloaders[phase]):
                
                image = Variable(image).to(device)
                attrs = Variable(attrs).to(device)
                batch_size = len(image)
                
                # pass data through model
                recon_image_1, recon_attrs_1, mu_1, logvar_1 = model(image, attrs, device=device)
                recon_image_2, recon_attrs_2, mu_2, logvar_2 = model(image, device=device)
                recon_image_3, recon_attrs_3, mu_3, logvar_3 = model(attrs=attrs, device=device)
                
                # compute ELBO for each data combo
                joint_loss = elbo_loss(recon_image_1, image, recon_attrs_1, attrs, mu_1, logvar_1, 
                                       lambda_image=1., lambda_attrs=0.01,
                                       annealing_factor=0.001)

                image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, 
                                       lambda_image=1., lambda_attrs=0.01,
                                       annealing_factor=0.001)

                attrs_loss = elbo_loss(None, None, recon_attrs_3, attrs, mu_3, logvar_3, 
                                       lambda_image=1., lambda_attrs=0.01,
                                       annealing_factor=0.001)

                JOINT_loss[epoch].append(joint_loss.item())
                IMG_loss[epoch].append(image_loss.item())
                ATTR_loss[epoch].append(attrs_loss.item())
                
                loss = joint_loss + image_loss + attrs_loss
                ELBOs[epoch].append(loss.item())

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

                    print("{}ep: {}: {}itr: loss={}, attr={}, img={}, joint={}, time={:.0f}m {:.0f}s".format(epoch+1, phase, i+1, 
                                                                              loss.item(), attrs_loss.item(),
                                                                              image_loss.item(), joint_loss.item(),
                                                                              time_elapsed//60, time_elapsed % 60))

                    df.loc[df.shape[0]] = [epoch+1, phase, i+1, get_lr(optimizer),
                                           np.mean(IMG_loss[epoch]), np.mean(ATTR_loss[epoch]),
                                           np.mean(JOINT_loss[epoch]), np.mean(ELBOs[epoch]),
                                           '{:.0f}m {:.0f}s'.format(time_elapsed // 60, int(time_elapsed % 60))]
                    df.to_csv(df_path, index=False)
                    
                if i % 100 == 99:
                    save_plot(model, image, attrs)
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
            print("FINALLY: {}ep: {}: {}itr: loss={}, attr={}, img={}, joint={}, time={:.0f}m {:.0f}s".format(epoch+1, 
                                                                              phase, i+1, 
                                                                              loss.item(), attrs_loss.item(),
                                                                              image_loss.item(), joint_loss.item(),
                                                                              time_elapsed//60, time_elapsed % 60))

            # Save logs
            df.loc[df.shape[0]] = [epoch+1, phase, i+1, get_lr(optimizer),
                                           np.mean(IMG_loss[epoch]), np.mean(ATTR_loss[epoch]),
                                           np.mean(JOINT_loss[epoch]), np.mean(ELBOs[epoch]),
                                           '{:.0f}m {:.0f}s'.format(time_elapsed // 60, int(time_elapsed % 60))]
            df.to_csv(df_path, index=False)
            save_plot(model, image, attrs)
            if phase == "train":
                model.train()
            
        scheduler.step()

