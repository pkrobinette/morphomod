import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tqdm
import time
from torch.utils.data import DataLoader
from evaluation import AverageMeter
import datasets as d
import json
from torch.onnx.symbolic_opset11 import hstack
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from models.gdino import GDINO
from PIL import Image
import os.path as osp
from torchvision import transforms
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None, help="location of dataset.")

    args = parser.parse_args()

    return args


def compute_f1_iou(mask, pred):
    """
    Both mask and pred are 2D arrays in {0,1}.
    If they are in [0,1] floats, threshold them first.
    """
    gt_flat  = mask.flatten()
    pred_flat = pred.flatten()

    f1  = f1_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat)  # Jaccard is effectively IoU

    return f1, iou


def main()
    checkpoint = "/content/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    gdino_model = GDINO()
    save_path = "/content/drive/MyDrive/HYDRA/SAM/auto_mask/CLWD/clwd_results.json"
    #
    # EVal
    #
    save_path = "/content/drive/MyDrive/HYDRA/SAM/gdino_bb/CLWD/clwd_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_root = os.path.dirname(save_path)
    os.makedirs(save_root, exist_ok=True)
    
    # make folders for images
    for folder in ["img", "mask", "gen_mask"]:
        os.makedirs(os.path.join(save_root, "_images", folder), exist_ok=True)
    
    if args.dataset == "clwd":
        dataloader = d.load_clwd("test", path='/content/data/CLWD', batch_size=1)
    elif args.dataset == "alpha1-S":
        dataloader = d.load_alpha1_dataset("test", path='/content/data/alpha1-S', batch_size=1)
    else:
        raise ValueError
    
    f1m = AverageMeter()
    ioum = AverageMeter()
    
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        img = np.array(batch['image'][0])
        mask = np.array(batch['mask'][0], dtype=np.uint8)
        imid = batch['id'][0]
    
        predictor.set_image(img)
        img_pil = Image.fromarray(img)
    
        input_box = gdino_model.predict([img_pil], ["watermark."])[0]
        input_box = np.array(input_box)
    
        pred_masks, scores, _ = predictor.predict(
          point_coords=None,
          point_labels=None,
          box=input_box[None, :],
          multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        pred_masks = np.array(pred_masks[sorted_ind][0], dtype=np.uint8)
    
        f1, iou = compute_f1_iou(mask, pred_masks)
    
        f1m.update(f1, 1)
        ioum.update(iou, 1)
        if imid < 100:
            # --- 4) Save images ---
            # Original image
            img_pil.save(osp.join(save_root, "_images", "img", f"{imid}.png"))
    
            # Ground-truth mask
            # If your mask is 0 or 255, you can save directly:
            mask_pil = mask * 255
            mask_pil = Image.fromarray(mask_pil)
            mask_pil.save(osp.join(save_root, "_images", "mask", f"{imid}.png"))
    
            # Generated (predicted) mask
            # Typically 0 or 1, so scale to 0..255 for viewing:
            gen_mask_pil = Image.fromarray((pred_masks * 255).astype(np.uint8))
            gen_mask_pil.save(osp.join(save_root, "_images", "gen_mask", f"{imid}.png"))
    
    results = {}
    results['f1'] = f1m.avg
    results['iou'] = ioum.avg
    
    with open(save_path, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()