import sys
sys.path.append("/content/hydra_steg/src")

from cnn import CNN
from steg_dataset import StegDisorientDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import os
import tqdm
import argparse
import utils
import segmentation_models_pytorch as smp
from diffusers import StableDiffusionInpaintPipeline
from models import Dilate
import os.path as osp
from datetime import datetime
from torchvision.utils import save_image
import json
import random


def get_args():
    """Get args for experiment"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_path", type=str, help="Where to save results.")
    parser.add_argument("--data_path", type=str, help="path of the dataset")
    parser.add_argument( 
        "--semseg_checkpoint", 
        type=str, 
        default=None,
        help="path to the semseg model checkpoint."
    )
    parser.add_argument( 
        "--class_checkpoint", 
        type=str, 
        default=None,
        help="path to the class model checkpoint."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-p", "--prompt", type=str, default="Remove.", help="The prompt to use for inpainting.")
    parser.add_argument("--dilate", type=int, default=0, help="dilation var.")
    parser.add_argument("--num_steps", type=int, default=50, help="number of inference steps during in-painting.")

    args = parser.parse_args()

    return args
    

def main(args):
    #
    # get datasets
    #
    test_dataset  = StegDisorientDataset(is_train='test',  root_dir=args.data_path)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)
    #
    # load all models
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # classification model
    class_model = CNN(num_classes=4).to(device)
    class_model = utils.load_checkpoint(class_model, args.class_checkpoint)
    # semseg model
    semseg_model = smp.Unet(
        encoder_name="mobilenet_v2",   
        encoder_weights="imagenet",
        in_channels=3,
        classes=1, 
    ).to(device)
    semseg_model = utils.load_checkpoint(semseg_model, args.semseg_checkpoint)
    # inpaint model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(device)
    #
    # set up saving dirs
    #
    utils.pprint_args(args)
    args.expr_path += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.expr_path, exist_ok=True)

    for folder in ["image", "mask", "gen_mask", "imfinal", "correct_bad"]:
        os.makedirs(osp.join(args.expr_path, "_images", folder))
    #
    # Eval
    # 
    class_model.eval()
    semseg_model.eval()
    transforms = T.ToTensor()
    
    correct = 0
    total = 0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask']
            imid = batch['id']

            # get orig class pred
            ypreds = class_model(imgs)

            # get semseg mask
            pred_masks = semseg_model(imgs)
            pred_masks = (pred_masks > 0.5).float().to(device)

            # dilate
            pred_masks = Dilate(pred_masks, 2).to(device)

            # inpatin
            xhat = pipe(
                prompt=["Remove."]*imgs.shape[0], 
                image=imgs, 
                mask_image=pred_masks,
            ).images
            # convert to tensors
            xhat_t = [transforms(im.resize(imgs[0].shape[1:])) for im in xhat]
            raw_out = torch.stack([t.detach() for t in xhat_t]).to(device)   
            imfinal = (raw_out * pred_masks) + ((1-pred_masks)*imgs)

            # disorient
            imfinal = disorient(imfinal, ypreds)

            #  classification
            outputs = class_model(imfinal.to(device))
            _, preds = outputs.max(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # saving stuff
            for im, m, genm, fin, i in zip(imgs, masks, pred_masks, imfinal, imid):
                save_image(m.float(), osp.join(args.expr_path, "_images", "mask", f"{i}.png"))
                save_image(genm.float(), osp.join(args.expr_path, "_images", "gen_mask", f"{i}.png"))
                save_image(im, osp.join(args.expr_path, "_images", "image", f"{i}.jpg"))
                save_image(fin, osp.join(args.expr_path, "_images", "imfinal", f"{i}.jpg"))
                cnt += 1

            for i, val in enumerate((preds == labels).int()):
                if val.item() == 1:
                    save_image(imfinal[i], osp.join(args.expr_path, "_images", "correct_bad", f"{imid[i]}.jpg"))

    results = {}
    results['acc'] = correct / total
    with open(os.path.join(args.expr_path, 'metrics.json'), 'w') as file:
        json.dump(results, file, indent=4)


def place_box(img, direction, box_size=50, margin=10, color=(1.0, 0.647, 0)):
    """
    Place a colored box on a single image tensor of shape (3, H, W).
    direction: one of ["up", "down", "left", "right"].
    color: (R, G, B) in [0..1] range for an orange box (e.g., (1, 0.647, 0)).
    """
    # Clone so we don't modify the original
    new_img = img.clone()
    _, H, W = new_img.shape  # channels, height, width

    # Determine box coordinates
    if direction == "up":
        rstart = margin
        rend = margin + box_size
        cstart = (W // 2) - (box_size // 2)
        cend = cstart + box_size
    elif direction == "down":
        rstart = H - margin - box_size
        rend = H - margin
        cstart = (W // 2) - (box_size // 2)
        cend = cstart + box_size
    elif direction == "left":
        cstart = margin
        cend = margin + box_size
        rstart = (H // 2) - (box_size // 2)
        rend = rstart + box_size
    elif direction == "right":
        cstart = W - margin - box_size
        cend = W - margin
        rstart = (H // 2) - (box_size // 2)
        rend = rstart + box_size
    else:
        # Default if something is off
        return new_img

    # Fill the box region with the chosen color
    new_img[0, rstart:rend, cstart:cend] = color[0]  # Red channel
    new_img[1, rstart:rend, cstart:cend] = color[1]  # Green channel
    new_img[2, rstart:rend, cstart:cend] = color[2]  # Blue channel

    return new_img


def disorient(images, raw_preds):
    """
    images:   torch.Tensor of shape (B, 3, H, W)
    raw_preds: model output logits of shape (B, num_classes)
    
    Each predicted direction is replaced by a random choice from avail_dirs,
    and an orange box is placed in that location on the image.
    """
    import random  # make sure to import if not already at the top

    # Directions that are "alternatives" for each class prediction
    avail_dirs = {
        0: ["left", "right", "down"],
        1: ["down", "left", "up"],
        2: ["left", "up", "right"],
        3: ["up", "right", "down"]
    }

    # Get predicted class for each image
    _, preds = raw_preds.max(dim=1)  # shape (B,)

    new_images = torch.zeros_like(images)  # same shape as images

    for i, (im, p) in enumerate(zip(images, preds)):
        # Pick a "new" direction at random
        direction = random.choice(avail_dirs[p.item()])
        # Place box on a clone of the original image
        new_images[i] = place_box(im, direction)

    return new_images



if __name__ == "__main__":
    args = get_args()
    main(args)
    
    