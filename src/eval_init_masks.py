"""
Script to evaluate initial mask models across each dataset.

**DATASETS TESTED (5):**
- CLWD
- LVW
- LOGO-Gray
- LOGO-mid
- LOGO-high

**MODELS TESTED (4):**
- slbr
- splitnet
- wdnet
- denet

**METRICS: **
- IoU
- Pixel Accuracy (MSE)
- Avg. TIME (s)

+ images for paper
"""
import argparse
import os
import time
from tqdm import tqdm
import torch
import pickle
import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    """Get args"""
    parser = argparse.ArgumentParser("Watermark Mask Test!")

    parser.add_argument('-d','--dataset', type=str, default='clwd', help="Dataset to use. Valid datasets = ['lvw', 'clwd', 'logo-g', 'logo-l', 'logo-h']")
    parser.add_argument('-m','--model', nargs='*', help="Model to evaluate. If None, all will be evaluated.['slbr', 'denet-g', 'denet-l', 'denet-h', 'wdnet', 'splitnet', slbr-ft]")
    parser.add_argument('--model_path', type=str, default='/Users/probinette/Documents/PROJECTS/hydra_steg/models')
    parser.add_argument('--data_path', type=str, default='/Users/probinette/Documents/PROJECTS/hydra_steg/data')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of the batch used during evaluation.')
    
    args = parser.parse_args()

    return args
    

def evaluate(models, args):
    """Evaluate models"""
    dataloader = utils.load_data(args)
    write_url = utils.get_save_name(args)

    all_results = {}

    for model_name, model in models.items():
        print(f"====> {model_name}")
        model.model.to(DEVICE)
        model.model.eval()
        iou_all = []
        mse_all = []
        time_all = []

        for i, batch in enumerate(tqdm(dataloader), 0):
            images = batch['image'].to(DEVICE)
            gt_masks = batch['mask'].to(DEVICE)
            #
            # predict masks
            #
            start = time.time()
            with torch.no_grad():  # Disable gradient computation for inference
                out_masks = model(images)
            total_time = time.time() - start
            #
            # calculate IoU
            #
            intersection = torch.logical_and(out_masks > 0.5, gt_masks > 0.5).float().sum((1, 2, 3))
            union = torch.logical_or(out_masks > 0.5, gt_masks > 0.5).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean().item()
            iou_all.append(iou)
            #
            # pixel accuracy
            #
            mse = torch.mean((out_masks - gt_masks) ** 2).item()
            mse_all.append(mse)
            #
            # record time
            #
            time_all.append(total_time)
        #
        # update all results
        #
        all_results[model_name] = {
            'iou_mean': sum(iou_all) / len(iou_all),
            'mse_mean': sum(mse_all) / len(mse_all),
            'time_mean': sum(time_all) / len(time_all),
        }
    #
    # save results --> update this
    #
    with open(write_url, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {write_url}")

    return all_results


def print_results(results):
    """Print results"""
    for model_name, r in results.items():
        print(f"Model: {model_name}\n")
        print(f"IoU Mean: {r['iou_mean']:.4f}\n")
        print(f"MSE Mean: {r['mse_mean']:.4f}\n")
        print(f"Time Mean: {r['time_mean']:.4f} seconds\n")
        print("-------------------\n")

    

if __name__ == "__main__":
    args = get_args()
    models = utils.get_models(args)
    all_results = evaluate(models, args)
    print_results(all_results)