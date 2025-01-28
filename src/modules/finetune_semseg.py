import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
from torchvision.utils import save_image
from collections import defaultdict
from PIL import Image
import torch
import os
import torch
from torchvision.utils import save_image
from sklearn.metrics import f1_score
from torchmetrics.functional import jaccard_index
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from evaluation import compute_IoU, FScore

torch.manual_seed(12)

class SemsegModule(pl.LightningModule):
    """
    Module for finetuning a segmentation model.

    Expects input batch to be a dictionary with at least these keys:
     - 'image'
     - 'target'
     - 'mask'
     - 'wm'
    """    
    def __init__(self, 
                model,         # model
                lr=0.001,      # Learning rate
                num_images_save=10, # number of images to save during testing
                save_path="test_outputs",
                num_classes=2,
                ):
        super().__init__()
        self.model = model
        self.lr= lr
        self.save_path = save_path
        self.num_images = num_images_save
        self.test_outputs = defaultdict(list)
        self.num_classes = num_classes
        
        self.first_batch = True

    def training_step(self, batch, _):
        inputs = batch['image']
        gt_masks = batch['mask'].float()
        #
        # predict
        #
        pred_masks = self.model(inputs)
        #
        # Calculate loss and log
        #
        loss = nn.functional.binary_cross_entropy(pred_masks, gt_masks)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['mask']
        #
        # Get output
        #
        prime_mask = self.model(images)
        prime_mask_pred = torch.where(prime_mask > 0.5, torch.ones_like(prime_mask), torch.zeros_like(prime_mask)).to(prime_mask.device)
        #
        # Save images -- Only for first batch
        #
        if self.first_batch:
            os.makedirs(self.save_path, exist_ok=True)
            # get random images
            indices = torch.randperm(images.shape[0])[:self.num_images]
            # save images
            for i, images in enumerate(zip(images[indices], gt_masks[indices], prime_mask_pred[indices])):
                in_im, gt_mask, pred_mask = images
                save_image(in_im, os.path.join(self.save_path, f'input_{i}.jpg'))
                save_image(gt_mask.float(), os.path.join(self.save_path, f'gt_mask_{i}.jpg'))
                save_image(pred_mask, os.path.join(self.save_path, f'pred_mask_{i}.jpg'))
            self.first_batch = False
        #
        # Get metrics (iou, f1 score, rmse) and log
        #
        iou = compute_IoU(prime_mask_pred, gt_masks)
        f1 = FScore(prime_mask_pred, gt_masks).item()
    
        metrics = {
            'iou': iou,
            'f1_score': f1,
        }
        
        self.log_dict(metrics)
    
        return metrics
                

    def on_test_epoch_end(self):
        avg_metrics = {metric: torch.stack(self.test_outputs[metric]).mean() for metric in self.test_outputs}
        for key, value in avg_metrics.items():
            self.log(f'{key}', value, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
