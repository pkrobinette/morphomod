"""
Image Quality metrics.
"""

import matlab.engine
import glob
import json
import argparse
from evaluation import AverageMeter
import os
import tqdm

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
        "--save_name", 
        type=str, 
        default=None, 
        help="where to save the results."
    )

    args = parser.parse_args()

    return args
    

def pprint(results, name):
    print("\n*"*50)
    print(" " * 20, name)

    for k,v in results.items():
        print(f"{k}: {v:.4f}")
        
    print("*"*50)


def run_eval(args):
    bval = AverageMeter()
    nval = AverageMeter()
    pval = AverageMeter()

    image_paths = glob.glob(os.path.join(args.data_path, "*.jpg"))

    for im_p in tqdm.tqdm(image_paths):
        im = eng.imread(im_p)
        bval.update(eng.brisque(im), 1)
        nval.update(eng.niqe(im), 1)
        pval.update(eng.piqe(im), 1)

    results = {}
    results['brisque'] = bval.avg
    results['niqe'] = nval.avg
    results['piqe'] = pval.avg

    with open(os.path.join(args.save_path, args.save_name + ".json"), "w") as f:
        json.dump(results, f, indent=4)

    pprint(results, args.save_name)

    return
    
    

if __name__ == "__main__":
    args = get_args()
    run_eval(args)