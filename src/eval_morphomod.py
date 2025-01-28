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
        default="clwd", 
        help="The dataset to use. ['clwd', '10kgray', '10khigh', '10kmid']"
    )
    parser.add_argument("--data_path", type=str, help="path of the dataset")
    parser.add_argument( 
        "--mask_checkpoint", 
        type=str, 
        default=None,
        help="path to the mask checkpoint."
    )
    parser.add_argument( 
        "--mask_refine_checkpoint", 
        type=str, 
        default=None,
        help="path to the mask refine model checkpoint."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--model_type", type=str, default='slbr', help="The model to finetune.")
    parser.add_argument("-p", "--prompt", type=str, default="Remove.", help="The prompt to use for inpainting.")
    parser.add_argument("--save", action="store_true", help="Indicate whether to save or not." )
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
        return datasets.load_clwd(is_train=is_train, batch_size=args.batch_size, path=args.data_path)
    if args.dataset == "10kgray":
        return datasets.load_10kgray(is_train=is_train, batch_size=args.batch_size, path=args.data_path)
    if args.dataset == "10kmid":
        return datasets.load_10kmid(is_train=is_train, batch_size=args.batch_size, path=args.data_path)
    if args.dataset == "10khigh":
        return datasets.load_10khigh(is_train=is_train, batch_size=args.batch_size, path=args.data_path)

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
    mask_model = utils.load_model(model_type=args.model_type, checkpoint=args.mask_checkpoint)
    mask_refine_model = models.UNetRefineSemseg()
    mask_refine_model = utils.load_checkpoint(mask_refine_model, args.mask_refine_checkpoint)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    )
    pipe.to(DEVICE)
    mask_refine_model.to(DEVICE)
    mask_model.to(DEVICE)
    #
    # Create folders for save
    #
    utils.pprint_args(args)
    args.expr_path += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.expr_path, exist_ok=True)

    if args.save:
        for folder in ["wm", "mask", "gen_mask", "refine_mask", "dilate_mask", "imfinal", "target"]:
            os.makedirs(osp.join(args.expr_path, "_images", folder))
    #
    # Run
    # 
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mask_refine_model.eval()
    mask_model.eval()
    print(f"==> testing {args.dataset}")
    rmses = AverageMeter()
    rmsews = AverageMeter()
    ssimesx = AverageMeter()
    psnresx = AverageMeter()
    maskIoU = AverageMeter()
    maskF1 = AverageMeter()
    lpipsesx = AverageMeter()
    lpip_lst = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader), 0):
            images = batch['image'].to(DEVICE)
            gt_masks = batch['mask'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            name_key = "targeturl" if args.dataset in ["10kgray", "10kmid", "10khigh"] else "img_path"
            new_names = [osp.basename(t) for t in batch[name_key]]

            # get mask
            gen_masks = mask_model(images)
            gen_masks = torch.where(
                gen_masks > 0.5, 
                torch.ones_like(gen_masks), 
                torch.zeros_like(gen_masks)
            ).to(DEVICE)
            # refine mask
            refine_masks = mask_refine_model(images, gen_masks)
            refine_masks = torch.where(
                refine_masks > 0.5,
                torch.ones_like(refine_masks),
                torch.zeros_like(refine_masks)).to("cuda")
            # dilate mask
            d_masks = models.Dilate(refine_masks, dilation_iterations=args.dilate)
            
            # INPAINT
            xhat = pipe(prompt=[args.prompt]*images.shape[0], image=images, mask_image=d_masks).images
            # convert to tensors

            xhat_t = [transform(im.resize(images[0].shape[1:])) for im in xhat]
            xhat_out = torch.stack([t.detach() for t in xhat_t]).to(DEVICE)
            imfinal = xhat_out * d_masks + (1-d_masks)*images
            imfinal.to(DEVICE)
            #
            # Calc Metrics
            # psnr, ssim, lpips, iou, f1
            #
            iou = compute_IoU(d_masks, gt_masks)
            f1 = FScore(d_masks, gt_masks).item()
            psnrx = 10 * log10(1 / F.mse_loss(imfinal,targets).item())       
            ssimx = pytorch_ssim.ssim(imfinal, targets).item()
            rmsex = compute_RMSE(imfinal, targets, gt_masks, is_w=False)
            rmsewx = compute_RMSE(imfinal, targets, gt_masks, is_w=True)

            im_lpip = norm(imfinal)
            tar_lpip = norm(targets)
            lpip = loss_fn_alex(im_lpip, tar_lpip).mean().item()
            # 
            # Update metrics
            #
            rmses.update(rmsex, images.size(0))
            rmsews.update(rmsewx, images.size(0))
            psnresx.update(psnrx, images.size(0))
            ssimesx.update(ssimx, images.size(0))
            maskIoU.update(iou)
            maskF1.update(f1, images.size(0))
            lpipsesx.update(lpip, images.size(0))

            if i % 100 == 0:
                print("Batch[%d]| PSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | maskIoU:%.4f | maskF1:%.4f | lpip:%.6f"
                %(i,psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, maskIoU.avg, maskF1.avg, lpipsesx.avg))
            # 
            # save images if indicated
            # ["wm", "mask", "gen_mask", "refine_mask", "dilate_mask", "imfinal", "target"]
            for im, m, gen_m, ref_m, dil_m, fin, tar, name in zip(images, gt_masks, gen_masks, refine_masks, d_masks, imfinal, targets, new_names):
                save_image(m.float(), osp.join(args.expr_path, "_images", "mask", name.replace(".jpg", ".png")))
                save_image(gen_m, osp.join(args.expr_path, "_images", "gen_mask", name.replace(".jpg", ".png")))
                save_image(ref_m, osp.join(args.expr_path, "_images", "refine_mask", name.replace(".jpg", ".png")))
                save_image(dil_m.float(), osp.join(args.expr_path, "_images", "dilate_mask", name.replace(".jpg", ".png")))
                save_image(im, osp.join(args.expr_path, "_images", "wm", name))
                save_image(fin, osp.join(args.expr_path, "_images", "imfinal", name))
                save_image(tar, osp.join(args.expr_path, "_images", "target", name))
        
    metrics = {
        "psnr": psnresx.avg,
        'ssim': ssimesx.avg,
        'rmse': rmses.avg,
        'rmsew': rmsews.avg,
        'mask_iou': maskIoU.avg,
        'mask_f1': maskF1.avg,
        'lpips': lpipsesx.avg
    }
    with open(os.path.join(args.expr_path, f'metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=4)
    print("Total:\nPSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | maskIoU:%.4f | maskF1:%.4f | lpip:%.6f "
                %(psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, maskIoU.avg, maskF1.avg, lpipsesx.avg))
    print("DONE.\n")


if __name__ == "__main__":
    args = get_args()
    main(args)