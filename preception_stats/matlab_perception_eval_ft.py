"""
Image Quality metrics. Specific to prompt datasets.

NIQE --> lower the better
BRISQUE --> lower the better
PIQE --> lower the better
"""

import matlab.engine
import glob
import json
import argparse
from evaluation import AverageMeter
import os
import tqdm
import tqdm
import cv2  # Assuming OpenCV for reading and cropping images
import numpy as np
import matplotlib.pyplot as plt
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None, 
        help="the data to eval"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
        help="where to save the results."
    )

    args = parser.parse_args()

    return args
    

def pprint(results, name):
    print("\n", "*"*50)
    print(" " * 20, name)

    for k,v in results.items():
        print(f"{k}: {v:.4f}")
        
    print("*"*50)


def run_eval(args):
    eng = matlab.engine.start_matlab()
    folders = glob.glob(os.path.join(args.data_path, "*"))
    folders = sorted([f for f in folders if os.path.isdir(f)])

    print("\n\nGathering metrics", "."*30)

    for folder in folders:
        save_name = os.path.basename(folder).split("_")[0]

        bval = AverageMeter()
        nval = AverageMeter()
        pval = AverageMeter()

        image_paths = glob.glob(os.path.join(folder, "_images", "imfinal", "*.jpg"))
        mask_paths = [os.path.join(folder, "_images", "mask", os.path.basename(f).replace("jpg", "png")) for f in image_paths]

        for im_p in tqdm.tqdm(image_paths):
            im = eng.imread(im_p)
            bval.update(eng.brisque(im), 1)
            nval.update(eng.niqe(im), 1)
            pval.update(eng.piqe(im), 1)

        # for im_p, m_p in tqdm.tqdm(zip(image_paths, mask_paths)):
        #     # load the image and mask
        #     im = cv2.imread(im_p)
        #     m = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
                        
        #     y_indices, x_indices = np.where(m == 255)
        #     if len(x_indices) > 0 and len(y_indices) > 0:
        #         x_min, x_max = x_indices.min(), x_indices.max()
        #         y_min, y_max = y_indices.min(), y_indices.max()
                
        #         cropped_im = im[y_min:y_max+1, x_min:x_max+1]
        #         cropped_im = np.ascontiguousarray(cropped_im)
                
        #         bval.update(eng.brisque(cropped_im), 1)
        #         nval.update(eng.niqe(cropped_im), 1)
        #         pval.update(eng.piqe(cropped_im), 1)
        #     else:
        #         print(f"No region with mask value 1 in {m_p}. Skipping.")
    
        results = {}
        results['brisque'] = bval.avg
        results['niqe'] = nval.avg
        results['piqe'] = pval.avg

        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(args.save_path, save_name + "_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)
    
        pprint(results, save_name)

    return
    

if __name__ == "__main__":
    args = get_args()
    run_eval(args)