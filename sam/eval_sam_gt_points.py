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

def get_bbox(mask: np.ndarray):
    """
    Given a 2D mask with values in [0,1], return the bounding box
    (x_min, y_min, x_max, y_max) of all non-zero pixels.

    If there are no non-zero pixels, return None.
    """
    rows, cols = np.where(mask > 0.5)

    if rows.size == 0 or cols.size == 0:
        return np.array([0, 0, 256, 256])

    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()

    return np.array([x_min, y_min, x_max, y_max])

def show_bbox_on_image(img: np.ndarray, bbox):
    """
    Display 'img' with a red rectangle bounding box.
    img can be 2D (grayscale) or 3D (H,W,3).
    bbox is a tuple (x_min, y_min, x_max, y_max).
    """
    if bbox is None:
        print("No bounding box (mask was empty).")
        plt.imshow(img, cmap="gray")
        plt.title("No bounding box found")
        plt.show()
        return

    x_min, y_min, x_max, y_max = bbox

    # Create a figure
    plt.figure()
    if img.ndim == 2:  # grayscale
        plt.imshow(img, cmap="gray")
    else:  # RGB
        plt.imshow(img)

    # Draw the bounding box as a red rectangle
    rect_width = x_max - x_min
    rect_height = y_max - y_min
    rect = plt.Rectangle(
        (x_min, y_min),
        rect_width,
        rect_height,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
    plt.gca().add_patch(rect)
    plt.title("Image with Bounding Box")
    plt.show()


def main()
    checkpoint = "/content/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    save_path = "/content/drive/MyDrive/HYDRA/SAM/auto_mask/CLWD/clwd_results.json"
    #
    # EVal
    #
    save_path = "/content/drive/MyDrive/HYDRA/SAM/gt_bb/CLWD/clwd_results.json"
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
        input_box = get_bbox(mask)
    
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
            img_pil = Image.fromarray(img)
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