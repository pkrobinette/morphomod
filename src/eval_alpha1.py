"""
Eval on the alpha1 dataset. 

Need to be able to select different types of models.
1. The OG models for each (--modeltype slbr, splitnet, slbr, etc.)
2. MorphoMod model. (--modeltype morphomod_slbr, morphomod_splitnet, etc.)
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
        return datasets.load_clwd(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "10kgray":
        return datasets.load_10kgray(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "10kmid":
        return datasets.load_10kmid(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "10khigh":
        return datasets.load_10khigh(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
    elif args.dataset == "alpha1-S":
        assert "alpha1-S" in args.data_path, "Whoops, did you mean alpha1-L?. Datapath does not match data selection."
        return datasets.load_alpha1_dataset(is_train=is_train, batch_size=args.batch_size, shuffle=False, path=args.data_path)
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
    # Load models
    #
    model = utils.load_model(model_type=args.model_type, checkpoint=args.checkpoint, refine_checkpoint=args.refine_checkpoint).to(DEVICE)
    #
    # Create folders for save
    #
    utils.pprint_args(args)
    args.expr_path += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.expr_path, exist_ok=True)

    for folder in ["wm", "mask", "gen_mask", "imfinal"]:
        os.makedirs(osp.join(args.expr_path, "_images", folder))
    #
    # Run
    # 
    rmsesw = AverageMeter()
    ssimsw = AverageMeter()
    psnrsw = AverageMeter()
    lpipsw = AverageMeter()
    rmsest = AverageMeter()
    ssimst = AverageMeter()
    psnrst = AverageMeter()
    lpipst = AverageMeter()
    
    epsilon = 1e-10
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    model.eval()
    cnt = 0
    
    for batch in tqdm.tqdm(dataloader):
        # load the images
        wm = batch['image'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        #
        # Calc imfinal
        #
        if "morphomod" in args.model_type:
            imfinal, immask = model(wm, args.dilate, args.prompt)
        elif args.model_type == "slbr":
            imoutput, immask_all, _ = model(wm)
            imoutput = imoutput[0] if type(imoutput) == list else imoutput
            immask = immask_all[0]
            imfinal = imoutput*immask + wm*(1-immask)
        elif args.model_type == "splitnet":
            imoutput,immask,_ = model(wm)
            imoutput = imoutput[0] if type(imoutput) == list else imoutput
            imfinal = imoutput*immask + wm*(1-immask)
        elif args.model_type == "wdnet":
            imfinal, immask,_,_,_ = model(wm)
        elif args.model_type in ["denet-g", "denet-l", "denet-h"]:
            imoutput,immask_all,_,_ = model(wm)
            imoutput = imoutput[0] if type(imoutput) == list else imoutput
            immask = immask_all[0]
            imfinal =imoutput*immask + wm*(1-immask)
        #
        # find the target metrics against true masks
        #
        imfinal_tr = imfinal * (1 - mask)
        wm_tr = wm * (1 - mask)

        mask_exp = mask.expand_as(imfinal)  # Broadcast the mask to [1, 3, 256, 256]
        zero_mask = mask_exp == 0
        imfinal_tr_pix = imfinal[zero_mask]
        wm_tr_pix = wm[zero_mask]

        with torch.no_grad():
          im_lpip = norm(imfinal_tr)
          tar_lpip = norm(wm_tr)
          lpiptx = loss_fn_alex(im_lpip, tar_lpip).mean().item()
          psnrtx = 10 * log10(1 / (F.mse_loss(imfinal_tr, wm_tr).item()+epsilon))
          ssimtx = pytorch_ssim.ssim(imfinal_tr, wm_tr).item()
          rmsetx = torch.sqrt(((imfinal_tr_pix - wm_tr_pix) ** 2).sum() / zero_mask.sum()).item()
        rmsest.update(rmsetx, wm.size(0))
        ssimst.update(ssimtx, wm.size(0))
        psnrst.update(psnrtx, wm.size(0))
        lpipst.update(lpiptx, wm.size(0))

        #
        # Get the W results
        #
        for w_im, imf_im, m_im in zip(wm, imfinal, mask):
          w_im = w_im.unsqueeze(0)
          imf_im = imf_im.unsqueeze(0)
          m_im = m_im.unsqueeze(0)
          non_zero_indices = torch.nonzero(m_im[0, 0], as_tuple=False)  # Only work on the first channel
          if non_zero_indices.numel() == 0:
            continue
          min_y, min_x = non_zero_indices.min(dim=0).values
          max_y, max_x = non_zero_indices.max(dim=0).values
          wm_cropped = w_im[:, :, min_y:max_y + 1, min_x:max_x + 1]
          imfinal_cropped = imf_im[:, :, min_y:max_y + 1, min_x:max_x + 1]
          mask_cropped = m_im[:, :, min_y:max_y + 1, min_x:max_x + 1]

          imfinal_wmr = imfinal_cropped * mask_cropped
          wm_wmr = wm_cropped * mask_cropped

          #
          # calculate all metrics
          #
          with torch.no_grad():
            # cropped metrics = [lpips, ssim, psnr]
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
            psnrwx = 10 * log10(1 / (F.mse_loss(imfinal_wmr, wm_wmr).item()+epsilon))
            ssimwx = pytorch_ssim.ssim(imfinal_wmr, wm_wmr).item()
            rmsewx = compute_RMSE(imfinal_wmr, wm_wmr, mask_cropped, is_w=True) / 256

            rmsesw.update(rmsewx, 1)
            ssimsw.update(ssimwx, 1)
            psnrsw.update(psnrwx, 1)
            lpipsw.update(lpipwx, 1)
        #
        # save 40 images
        #
        if cnt < 40:
            for im, m, gen_m, fin in zip(wm, mask, immask, imfinal):
                save_image(m.float(), osp.join(args.expr_path, "_images", "mask", f"{cnt}.png"))
                save_image(gen_m.float(), osp.join(args.expr_path, "_images", "gen_mask", f"{cnt}.png"))
                save_image(im, osp.join(args.expr_path, "_images", "wm", f"{cnt}.jpg"))
                save_image(fin, osp.join(args.expr_path, "_images", "imfinal", f"{cnt}.jpg"))
                cnt += 1

    # record resutls
    results = {}
    results["rmsew"] = rmsesw.avg
    results["ssimw"] = ssimsw.avg
    results["psnrw"] = psnrsw.avg
    results["lpipw"] = lpipsw.avg
    results["rmset"] = rmsest.avg
    results["ssimt"] = ssimst.avg
    results["psnrt"] = psnrst.avg
    results["lpipt"] = lpipst.avg
    results["prompt"] = args.prompt
    results["dilate"] = args.dilate

    # save everything
    with open(os.path.join(args.expr_path, f'{args.dataset}_{args.model_type}_metrics.json'), 'w') as file:
      json.dump(results, file, indent=4)

    print(results)


if __name__ == "__main__":
    args = get_args()
    main(args)