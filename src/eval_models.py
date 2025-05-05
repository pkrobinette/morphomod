"""
Eval the number of inference steps during generation.
"""

import os.path as osp
import os
import sys
from datetime import datetime
import datasets
import utils
import models
from diffusers import StableDiffusionInpaintPipeline
import argparse
import torch
from torchvision import transforms
import tqdm
from evaluation import compute_IoU, FScore, compute_RMSE, AverageMeter
import torch.nn.functional as F
from math import log10
from metrics import pytorch_ssim
import lpips
import json
from torchvision.utils import save_image
from models import Dilate
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


def get_args():
    """Get args for experiment"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_path", type=str, help="Where to save results.")
    parser.add_argument(
        "-d", 
        "--dataset", 
        type=str, 
        default="alpha1-S", 
        help="The dataset to use. ['alpha1-S', 'alpha1-L', 'CLWD', '10kgray', '10khigh', '10kmid']"
    )
    parser.add_argument("--data_path", type=str, help="path of the dataset")
    parser.add_argument( 
        "--checkpoint", 
        type=str, 
        default=None,
        help="path to the mask checkpoint."
    )
    parser.add_argument( 
        "--refine_checkpoint", 
        type=str, 
        default=None,
        help="path to the mask refine model checkpoint."
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--model_type", type=str, default='slbr', help="The model to finetune.")
    parser.add_argument("-p", "--prompt", type=str, default="Remove.", help="The prompt to use for inpainting.")
    parser.add_argument("--dilate", type=int, default=0, help="dilation var.")
    parser.add_argument("--fill", type=str, default=None, help="['black', 'background']==> How to fill in the wm prior to generation.")
    parser.add_argument("--num_steps", type=int, default=50, help="number of inference steps during in-painting.")
    parser.add_argument("--inpaint_mod", type=str, default='SD2', help="The inpainting model to use.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples to evaluate.")

    args = parser.parse_args()

    return args

def get_dataset(is_train: str, args: argparse.Namespace):
    """Load dataset.

    Args:
        dataset: key for the dataset
        is_train: "test" or "train"
    Returns:
        DataLoader
    """
    if args.dataset == "CLWD":
        return datasets.load_clwd(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path, num_samples=args.num_samples)
    elif args.dataset == "10kgray":
        return datasets.load_10kgray(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "10kmid":
        return datasets.load_10kmid(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "10khigh":
        return datasets.load_10khigh(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "alpha1-S":
        assert "alpha1-S" in args.data_path, "Whoops, did you mean alpha1-L?. Datapath does not match data selection."
        return datasets.load_alpha1_dataset(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path, num_samples=args.num_samples)
    elif args.dataset == "alpha1-L":
        assert "alpha1-L" in args.data_path, "Whoops, did you mean alpha1-S?. Datapath does not match data selection."
        return datasets.load_alpha1_dataset(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)

    raise ValueError("Dataset does not exist.")

def norm(input):
    max_val = torch.max(input)
    min_val = torch.min(input)
    input = (input-min_val)/(max_val-min_val)
    input = 2*input - 1
    return input

def main(args):
    #
    # load dataset
    #
    dataloader = get_dataset("test", args)
    #
    # load model
    #
    model = utils.load_model(
        model_type=args.model_type, 
        checkpoint=args.checkpoint, 
        refine_checkpoint=args.refine_checkpoint,
        inpaint=args.inpaint_mod
    ).to(DEVICE)
    #
    # create folders
    #
    utils.pprint_args(args)
    args.expr_path += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.expr_path, exist_ok=True)

    for folder in ["wm", "mask", "gen_mask", "imfinal", "raw_out", "fill_x"]:
        os.makedirs(osp.join(args.expr_path, "_images", folder))
    #
    # metrics
    #
    rmsesw = AverageMeter()
    ssimsw = AverageMeter()
    psnrsw = AverageMeter()
    lpipsw = AverageMeter()

    epsilon = 1e-10
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    model.eval()
    cnt = 0

    for batch in tqdm.tqdm(dataloader):
        wm = batch['image'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        ids = batch['id']

        if "morphomod" in args.model_type:
            with torch.no_grad():
                imfinal, immask, raw_out, fill_x = model.forward_w_gt(
                    wm, mask, args.dilate, args.prompt,
                    fill=args.fill, num_steps=args.num_steps
                )
        else:
            raise ValueError("whoops wrong model.")

        for w_im, imf_im, m_im in zip(wm, imfinal, mask):
            w_im = w_im.unsqueeze(0)
            imf_im = imf_im.unsqueeze(0)
            m_im = m_im.unsqueeze(0)
            non_zero_indices = torch.nonzero(m_im[0, 0], as_tuple=False)
            if non_zero_indices.numel() == 0:
                continue
            min_y, min_x = non_zero_indices.min(dim=0).values
            max_y, max_x = non_zero_indices.max(dim=0).values
            wm_cropped = w_im[:, :, min_y:max_y + 1, min_x:max_x + 1]
            imfinal_cropped = imf_im[:, :, min_y:max_y + 1, min_x:max_x + 1]
            mask_cropped = m_im[:, :, min_y:max_y + 1, min_x:max_x + 1]

            imfinal_wmr = imfinal_cropped * mask_cropped
            wm_wmr = wm_cropped * mask_cropped

            with torch.no_grad():
                im_lpip = norm(imfinal_wmr)
                tar_lpip = norm(wm_wmr)
                try:
                    lpipwx = loss_fn_alex(im_lpip, tar_lpip).mean().item()
                except:
                    min_size = 32
                    newh = max(min_size, im_lpip.shape[2])
                    neww = max(min_size, im_lpip.shape[3])
                    im_lpip = F.interpolate(im_lpip, size=(newh, neww), mode='bilinear', align_corners=False)
                    tar_lpip = F.interpolate(tar_lpip, size=(newh, neww), mode='bilinear', align_corners=False)
                    lpipwx = loss_fn_alex(im_lpip, tar_lpip).mean().item()

                psnrwx = 10 * log10(1 / (F.mse_loss(imfinal_wmr, wm_wmr).item() + epsilon))
                ssimwx = pytorch_ssim.ssim(imfinal_wmr, wm_wmr).item()
                rmsewx = compute_RMSE(imfinal_wmr, wm_wmr, mask_cropped, is_w=True) / 256

                rmsesw.update(rmsewx, 1)
                ssimsw.update(ssimwx, 1)
                psnrsw.update(psnrwx, 1)
                lpipsw.update(lpipwx, 1)

        for im, m, gen_m, fin, raw, fx, imid in zip(wm, mask, immask, imfinal, raw_out, fill_x, ids):
            save_image(m.float(), osp.join(args.expr_path, "_images", "mask", f"{imid}.png"))
            save_image(gen_m.float(), osp.join(args.expr_path, "_images", "gen_mask", f"{imid}.png"))
            save_image(im, osp.join(args.expr_path, "_images", "wm", f"{imid}.jpg"))
            save_image(fin, osp.join(args.expr_path, "_images", "imfinal", f"{imid}.jpg"))
            save_image(raw, osp.join(args.expr_path, "_images", "raw_out", f"{imid}.jpg"))
            save_image(fx, osp.join(args.expr_path, "_images", "fill_x", f"{imid}.jpg"))
            cnt += 1
        #
        # release memory
        #
        del wm, mask, imfinal, immask, raw_out, fill_x, batch, w_im, imf_im, m_im
        del wm_cropped, imfinal_cropped, mask_cropped
        del imfinal_wmr, wm_wmr, im_lpip, tar_lpip
        torch.cuda.empty_cache()
        gc.collect()
    #
    # save results
    #
    results = {
        "rmsew": rmsesw.avg,
        "ssimw": ssimsw.avg,
        "psnrw": psnrsw.avg,
        "lpipw": lpipsw.avg,
        "prompt": args.prompt,
        "dilate": args.dilate,
        "fill": args.fill,
    }

    with open(os.path.join(args.expr_path, f'{args.dataset}_{args.inpaint_mod}_metrics.json'), 'w') as file:
        json.dump(results, file, indent=4)

    print(results)



if __name__ == "__main__":
    args = get_args()
    main(args)