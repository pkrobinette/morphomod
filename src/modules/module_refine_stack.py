import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from collections import defaultdict
from math import log10
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from evaluation import compute_RMSE, FScore
from metrics import pytorch_ssim
from segmentation_models_pytorch import Unet

torch.manual_seed(12)

class RefineModuleStack(pl.LightningModule):
    """
    Module for refining images with an image and mask as input.

    Expects input batch to be a dictionary with at least these keys:
     - 'image': The input image
     - 'mask': The corresponding mask
     - 'target': The ground truth image
    """    
    def __init__(self, 
                 model,         # Model instance
                 lr=0.001,      # Learning rate
                 num_images_save=10, # Number of images to save during testing
                 save_path="test_outputs"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_path = save_path
        self.num_images = num_images_save
        self.test_outputs = defaultdict(list)
        self.first_batch = True

    def training_step(self, batch, _):
        images = batch['image']
        masks = batch['mask']
        targets = batch['target']
                
        # Predict
        xhat = self.model(images, masks)
        
        # Calculate loss
        ssim_loss = pytorch_ssim.ssim(xhat, targets)
        loss = F.mse_loss(xhat, targets) + (1 - ssim_loss)
    
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        targets = batch['target']
        
        # Predict
        xhat = self.model(images, masks)
        
        # Save images -- Only for the first batch
        if self.first_batch:
            os.makedirs(self.save_path, exist_ok=True)
            indices = torch.randperm(images.shape[0])[:self.num_images]
            for i, img_set in enumerate(zip(images[indices], targets[indices], xhat[indices])):
                in_im, gt_im, xout_im = img_set
                save_image(in_im, os.path.join(self.save_path, f'input_{i}.jpg'))
                save_image(gt_im.float(), os.path.join(self.save_path, f'target_{i}.jpg'))
                save_image(xout_im, os.path.join(self.save_path, f'xout_{i}.jpg'))
            self.first_batch = False
        
        # Calculate metrics
        psnrx = 10 * log10(1 / F.mse_loss(xhat, targets).item())
        ssimx = pytorch_ssim.ssim(xhat, targets)
        
        # Calculate RMSE and Weighted RMSE
        rmsex = compute_RMSE(xhat, targets, masks, is_w=False)
        rmsewx = compute_RMSE(xhat, targets, masks, is_w=True)
    
        metrics = {
            'psnrx': psnrx,
            'ssimx': ssimx,
            'rmsex': rmsex,
            'rmsewx': rmsewx
        }
        
        self.log_dict(metrics)
    
        return metrics

    def on_test_epoch_end(self):
        avg_metrics = {metric: torch.stack(self.test_outputs[metric]).mean() for metric in self.test_outputs}
        for key, value in avg_metrics.items():
            self.log(f'{key}', value, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
