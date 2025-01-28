from PIL import Image
import numpy as np
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms as T
import random
import glob

class DilateDataset(torch.utils.data.Dataset):
    def __init__(self, dilate_val: int, path: str):
        all_folders = sorted(glob.glob(osp.join(path, "**")))
        d_folder = [f for f in all_folders if f"dilate_{dilate_val}" in f][-1]

        self.root_main = osp.join(path, d_folder, "_images")

        self.wm_path=osp.join(self.root_main,'wm','%s.jpg')
        self.imfinal_path=osp.join(self.root_main,'imfinal','%s.jpg')
        self.mask_path=osp.join(self.root_main,'mask','%s.png')
        
        self.ids = list()
        for file in os.listdir(osp.join(self.root_main, 'wm')):
            self.ids.append(file.strip('.jpg'))
            
        self.transform_tensor = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        print("\n\n==> Num images: ", len(self.ids))
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        imid = self.ids[index]

        # load images
        wm = Image.open(self.wm_path%imid)
        imfinal = Image.open(self.imfinal_path%imid)
        mask = Image.open(self.mask_path%imid).convert("L")

        wm = self.transform_tensor(wm)
        imfinal = self.transform_tensor(imfinal)
        mask = self.transform_tensor(mask)
        mask = (mask > 0.5).float()

        data = {
            'wm': wm,
            'imfinal': imfinal,
            'mask': mask,
        }

        return data
