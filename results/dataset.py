import os
import sys
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

ATTR_TO_IX_DICT  = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18, 
                    'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38, 
                    'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21, 
                    'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26, 
                    'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32, 
                    'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22, 
                    'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3, 
                    'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23, 
                    'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}

ATTR_IX_TO_KEEP  = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT  = {v:k for k, v in ATTR_TO_IX_DICT.items()}
N_ATTRS          = len(ATTR_IX_TO_KEEP)


def get_attr(img_name, df_attr):
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

class FaceData_with_Attributes(Dataset):
    def __init__(self, img_names, img_path, df_attr_path, image_transform=None, attr_transform=None):

        self.img_path = img_path
        self.df_attr = pd.read_csv(df_attr_path)
        
        self.img_names = img_names

        self.size = int(len(img_names))

        self.image_transform = image_transform
        self.attr_transform  = attr_transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):

        # attr
        attr = get_attr(self.img_names[index], self.df_attr)

        if self.attr_transform is not None:
            attr = self.attr_transform(attr)

        # img 
        img_path = os.path.join(self.img_path, self.img_names[index])

        img = Image.open(img_path)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = img.crop((32, 32, 224, 224))
        img = img.resize((64, 64), Image.ANTIALIAS)
        if self.image_transform is not None:
            img  = self.image_transform(img)

        img = np.array(img)

        return img.transpose(2, 0, 1) / 255., attr
#         return img, attr

    def __len__(self):
        return self.size