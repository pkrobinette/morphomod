from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
from .base_dataset import get_transform
import random

class Refine10kDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args):
        
        args.is_train = is_train == 'train'
        self.train = args.is_train
        if args.is_train == True:
            print("TRAIN")
            self.root_main = osp.join(args.dataset_dir, 'train')
            self.keep_background_prob = -1
        elif args.is_train == False:
            self.root_main = osp.join(args.dataset_dir, 'test')
            self.keep_background_prob = -1
            args.preprocess = 'resize'
            args.no_flip = True

        self.args = args
        # Augmentataion?
        self.transform_norm=transforms.Compose([transforms.ToTensor()])
        self.augment_transform = get_transform(args, 
            additional_targets={'W':'image', 'T':'image', 'In':'image', 'mask':'mask', 'gen_mask':'mask' }) #,
        self.transform_tensor = transforms.ToTensor()

        self.imageW_path=osp.join(self.root_main,'watermarked','%s.jpg')
        self.imageT_path=osp.join(self.root_main,'target','%s.jpg')
        self.imageIn_path=osp.join(self.root_main,'inpainted','%s.jpg')
        self.mask_path=osp.join(self.root_main,'mask','%s.png')
        self.gen_mask_path=osp.join(self.root_main,'gen_mask','%s.png')
		
        self.ids = list()
        for file in os.listdir(osp.join(self.root_main, 'target')):
            self.ids.append(file.strip('.jpg'))
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        
    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, index):
        img_id = self.ids[index]

        # watermarked image
        img_W = cv2.imread(self.imageW_path%img_id)
        img_W = cv2.cvtColor(img_W, cv2.COLOR_BGR2RGB)

        # target images
        img_T = cv2.imread(self.imageT_path%img_id)
        img_T = cv2.cvtColor(img_T, cv2.COLOR_BGR2RGB)

        # Inpainted images
        img_In = cv2.imread(self.imageIn_path%img_id)
        img_In = cv2.cvtColor(img_In, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_path%img_id)
        mask = mask[:, :, 0].astype(np.float32) / 255.

        gen_mask = cv2.imread(self.gen_mask_path%img_id)
        gen_mask = gen_mask[:, :, 0].astype(np.float32) / 255.
                
        return {'W': img_W, 'T': img_T, 'In': img_In, 'mask':mask, 'gen_mask':gen_mask, 'img_path':self.imageW_path%img_id}


    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        W = self.transform_norm(sample['W'])
        T = self.transform_norm(sample['T'])
        In = self.transform_norm(sample['In'])

        mask = sample['mask'][np.newaxis, ...].astype(np.float32)
        mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)

        gen_mask = sample['gen_mask'][np.newaxis, ...].astype(np.float32)
        gen_mask = np.where(gen_mask > 0.1, 1, 0).astype(np.uint8)
        
        data = {
            'image': W,
            'target': T,
            'inpainted': In,
            'mask': mask,
            'gen_mask':gen_mask,
            'img_path':sample['img_path']
        }

        return data

    def check_sample_types(self, sample):
        assert sample['W'].dtype == 'uint8'
        assert sample['T'].dtype == 'uint8'
        assert sample['In'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.augment_transform is None:
            return sample
        #print(self.transform.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augment_transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augment_transform(image=sample['T'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            #print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 100



