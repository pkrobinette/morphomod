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
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="which model to eval."
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

    print(folders)

    if args.model == "SDXL":
        folder = [f for f in folders if "SDXL" in f][0]
    elif args.model == "LaMa":
        folder = [f for f in folders if "LaMa" in f][0]

    print("\n\nGathering metrics", "."*30)


    bval = AverageMeter()
    nval = AverageMeter()
    pval = AverageMeter()

    if args.model == "SDXL":
        image_paths = glob.glob(os.path.join(folder, "_images", "imfinal", "*.jpg"))
    else:
        image_paths = glob.glob(os.path.join(folder, "*.png"))
        
    for im_p in tqdm.tqdm(image_paths):
        im = eng.imread(im_p)
        bval.update(eng.brisque(im), 1)
        nval.update(eng.niqe(im), 1)
        pval.update(eng.piqe(im), 1)

    results = {}
    results['brisque'] = bval.avg
    results['niqe'] = nval.avg
    results['piqe'] = pval.avg
    
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, args.model + "_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    pprint(results, args.model)

    return
    

if __name__ == "__main__":
    args = get_args()
    run_eval(args)