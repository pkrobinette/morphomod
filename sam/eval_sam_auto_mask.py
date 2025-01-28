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
from torch.onnx.symbolic_opset11 import hstack
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import json
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
    sam2_checkpoint = "/content/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    save_path = "/content/drive/MyDrive/HYDRA/SAM/auto_mask/CLWD/clwd_results.json"
    #
    # EVal
    #
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
        pred_masks = mask_generator.generate(img)
    
        # get best metric
    
        max_iou = 0
        max_f1 = 0
        max_id = 0
    
        for i in range(len(pred_masks)):
          pred = np.array(pred_masks[i]['segmentation'], dtype=np.uint8)
          f1, iou = compute_f1_iou(mask, pred)
    
          if iou > max_iou:
            max_id = i
            max_iou = iou
            max_f1 = f1
    
        f1m.update(max_f1, 1)
        ioum.update(max_iou, 1)
    
        try:
            pred = np.array(pred_masks[max_id]['segmentation'], dtype=np.uint8)
            if imid < 100:
                # --- 4) Save images ---
                # Original image
                img_pil = Image.fromarray(img)
                img_pil.save(osp.join(save_root, "_images", "img", f"{imid}.png"))
    
                # Ground-truth mask
                # If your mask is 0 or 255, you can save directly:
                mask_pil = Image.fromarray(mask)  # shape: (H, W)
                mask_pil.save(osp.join(save_root, "_images", "mask", f"{imid}.png"))
    
                # Generated (predicted) mask
                # Typically 0 or 1, so scale to 0..255 for viewing:
                gen_mask_pil = Image.fromarray((pred * 255).astype(np.uint8))
                gen_mask_pil.save(osp.join(save_root, "_images", "gen_mask", f"{imid}.png"))
        except:
            continue
    
    results = {}
    results['f1'] = f1m.avg
    results['iou'] = ioum.avg
    
    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()