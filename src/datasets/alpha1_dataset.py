from PIL import Image
import numpy as np
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms as T
import random
import glob

class AlphaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train: str, path: str, num_samples = None):

        is_train = is_train == "train"
        if is_train:
            self.root_main = os.path.join(path, "train")
        else:
            self.root_main = os.path.join(path, "test")

        self.image_path=osp.join(self.root_main,'image','%s.jpg')
        self.mask_path=osp.join(self.root_main,'mask','%s.png')
        
        self.ids = list()
        for file in os.listdir(osp.join(self.root_main, 'image')):
            self.ids.append(int(file.strip('.jpg')))
            
        self.transform_tensor = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        #
        # sort images
        #
        self.ids.sort()
        if num_samples:
            self.ids = self.ids[:num_samples]

        print("\n\n==> Num images: ", len(self.ids))
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        imid = self.ids[index]

        # load images
        image = Image.open(self.image_path%imid)
        mask = Image.open(self.mask_path%imid).convert("L")

        image = self.transform_tensor(image)
        mask = self.transform_tensor(mask)
        mask = (mask > 0.5).float()

        data = {
            'image': image,
            'mask': mask,
            'id': imid
        }

        return data
