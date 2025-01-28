import logging
import torch
from PIL import Image
import numpy as np
from typing import Union


def get_device_type() -> str:
    """Get the device type."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("No GPU found, using CPU instead")
        return "cpu"


def make_mask_from_bounding_box(img, bbox) -> Image.Image:
    """Make a segmentation mask from a bounding box

    Args:
        img: the original image.
        bbox: the bounding box (x_min, y_min, x_max, y_max)

    Returns:
        binary mask where the bbox is 255 and everything else is 0.
    """
    h, w = img.size
    # get x,y values from bbox
    x_min, y_min, x_max, y_max = bbox
    box_mask = np.zeros((w,h)) # have to flip

    # update the mask to be 255 for the bounding box
    for r in range(box_mask.shape[0]):
        for c in range(box_mask.shape[1]):
            if r >= y_min and r <= y_max and c >= x_min and c <= x_max:
              box_mask[r][c] = 255

    return box_mask

def convert_np_to_pil(img: Union[list[np.array], np.array]):
    """Convert numpy array to PIL Image."""
    if type(img) == list:
        return [convert_np_to_pil(i) for i in img]

    if int(img.max()) <= 1:
        img *= 255 # put image in right range

    while len(img.shape) > 2:
        img = img[0]

    return Image.fromarray(img)
    