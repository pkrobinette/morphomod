import numpy as np
from PIL import Image
import cv2  # Required for SSIM
from skimage.metrics import structural_similarity as ssim


def image_to_array(image):
    """Convert a PIL Image to a normalized NumPy array."""
    return np.array(image).astype(np.float32) / 255.0


def calc_psnr(original, modified):
    """Calculate the PSNR between two images.

    Args:
        original (Image.Image): The original image.
        modified (Image.Image): The modified image.
    
    Returns:
        float: The PSNR value.
    """
    original_array = image_to_array(original)
    modified_array = image_to_array(modified)
    mse = np.mean((original_array - modified_array) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def calc_ssim(original, modified, data_range=1.0):
    """Calculate the SSIM between two images.

    Args:
        original (Image.Image): The original image.
        modified (Image.Image): The modified image.
        data_range (float): The data range of the input image. Set to 1.0 if images are normalized to [0, 1].
    
    Returns:
        float: The SSIM value.
    """
    original_array = image_to_array(original)
    modified_array = image_to_array(modified)
    
    # Convert to grayscale if images are RGB
    if original_array.ndim == 3:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        modified_array = cv2.cvtColor(modified_array, cv2.COLOR_RGB2GRAY)
    
    ssim_value, _ = ssim(original_array, modified_array, data_range=data_range, full=True)
    return ssim_value


def calc_rmse(original, modified):
    """Calculate the RMSE between two images.

    Args:
        original (Image.Image): The original image.
        modified (Image.Image): The modified image.

    Returns:
        float: The RMSE value.
    """
    original_array = image_to_array(original)
    modified_array = image_to_array(modified)
    mse = np.mean((original_array - modified_array) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calc_rmse_w(original, modified, mask):
    """Calculate the RMSE in the watermarked region only.

    Args:
        original (Image.Image): The original image.
        modified (Image.Image): The modified image.
        mask (Image.Image): The binary mask of the watermarked region.
    
    Returns:
        float: The RMSE in the watermarked region.
    """
    original_array = image_to_array(original)
    modified_array = image_to_array(modified)
    mask_array = image_to_array(mask)  # Assuming mask is binary (0 and 1)
    
    diff = (original_array - modified_array) ** 2
    masked_diff = diff * mask_array
    mse_w = np.sum(masked_diff) / np.sum(mask_array)
    rmse_w = np.sqrt(mse_w)
    return rmse_w
