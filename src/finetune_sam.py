"""
Fine-tune the SAM2 model on a custom dataset.
"""

import numpy as np
import torch
import cv2
import os
from torch.onnx.symbolic_opset11 import hstack
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import argparse


def get_args():
    """Parse arguments for the script"""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--data_path', type=str, required=True)
    argument_parser.add_argument('--save_path', type=str, required=True)

    args = argument_parser.parse_args()
    return args


def get_data(data_dir: str):
    """Get the data from the data directory
    
    Args:
        data_dir: the path to the data directory

    Returns:
        data: a list of dictionaries containing the path to the image and the
            >> {"image": str, "annotation": str}
    """
    data = []
    for name in os.listdir(os.path.join(data_dir, "image")):
        data.append({"image": os.path.join(data_dir, "image", name), "annotation": os.path.join(data_dir, "mask", name)})

    return data


def get_bounding_box(mask: np.array):
    """Get the bounding box of the mask"""
    coords = np.argwhere(mask > 0)  # Get all coordinates in the binary mask where the mask is 1
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    return [min_x, min_y, max_x, max_y]


def read_single(data: dict, bbox_noise: int = 300):
    """Read a single image and its annotation from the data."""
    #  select image
    data_pt  = data[np.random.randint(len(data))] # choose random entry
    img = cv2.imread(data_pt["image"])[...,::-1]  # read image
    mask = cv2.imread(data_pt["annotation"], cv2.IMREAD_GRAYSCALE) # read annotation

    # resize image
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]]) # scalling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
    if img.shape[0]<1024:
        img = np.concatenate([img,np.zeros([1024 - img.shape[0], img.shape[1],3],dtype=np.uint8)],axis=0)
        mask = np.concatenate([mask, np.zeros([1024 - mask.shape[0], mask.shape[1]], dtype=np.uint8)],axis=0)
    if img.shape[1]<1024:
        img = np.concatenate([img, np.zeros([img.shape[0] , 1024 - img.shape[1], 3], dtype=np.uint8)],axis=1)
        mask = np.concatenate([mask, np.zeros([mask.shape[0] , 1024 - mask.shape[1]], dtype=np.uint8)],axis=1)

    # get a bounding box of the mask and add random noise
    bbox = get_bounding_box(mask)
    for i in range(2):
        bbox[i] = bbox[i] - np.random.randint(bbox_noise)
    for i in range(2,4):
        bbox[i] = bbox[i] + np.random.randint(bbox_noise)
    return img, mask,bbox


def read_batch(data: dict, batch_size: int = 4):
    """Read a batch of images and their annotations from the data."""
    limage = []
    lmask = []
    linput_bbox = []
    for _ in range(batch_size):
        image,mask,bbox = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_bbox.append(bbox)

    return limage, np.array(lmask), np.array(linput_bbox),  np.ones([batch_size,1])


def finetune(predictor, data, model_path, image_path):
    """Finetune the model"""
    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
    predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # mixed precision
    #
    # Finetune
    #
    for itr in range(100000):
        with torch.cuda.amp.autocast(): # cast to mix precision
            image,mask,input_box, input_label = read_batch(data, batch_size=4) # load data batch
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image_batch(image) # apply SAM image encoder to the image

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point_coords=None, point_labels=input_label, box=input_box, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=None,boxes=unnorm_box,masks=None,)
            #
            # mask decoder
            #
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=False,repeat_image=False,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution
            #
            # Segmentaion Loss caclulation
            #
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss
            #
            # Score loss calculation (intersection over union) IOU
            #
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses
            #
            # apply back propogation
            #
            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision
            #
            # Save model and training images
            #
            if itr%1000==0: 
                torch.save(predictor.model.state_dict(), os.path.join(model_path, "model.torch")) # save model
                fig, ax = plt.subplots(4, 2, figsize=(5,10))

                for i in range(4):
                    ax[i, 0].imshow(gt_mask[i].cpu().numpy())
                    ax[i, 0].set_axis_off()
                    ax[i, 1].imshow(prd_mask[i].detach().cpu().numpy())
                    ax[i, 1].set_axis_off()
                plt.savefig(os.path.join(image_path, f"masks_comparison_step-{itr}.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)  # Close the figure after saving
            #
            # Display results
            #
            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            if itr%100==0: print("step)",itr, "Accuracy(IOU)=",mean_iou)
    
    return


def main():
    """Main function."""
    args = get_args()
    #
    # Make save folders
    #
    image_path = os.path.join(args.save_path, "training_images")
    model_path = os.path.join(args.save_path, "checkpoints")
    os.makedirs(args.save_path, exist_ok = True)
    os.makedirs(image_path, exist_ok = True)
    os.makedirs(model_path, exist_ok = True)
    #
    # Get data and checkpoints
    #
    data = get_data(args.data_path)
    checkpoint = "/content/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    #
    # fintune the model
    #
    finetune(predictor, data, model_path, image_path)


if __name__ == "__main__":
    main()
