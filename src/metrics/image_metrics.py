"""
Image quality metrics for stegomarking removal.
"""

import torch
import torch.nn.functional as F


def calc_psnr(original, modified):
    """Calculate the PSNR between two batches of images.

    Args:
        original (torch.Tensor): The original images with shape (B, C, H, W).
        modified (torch.Tensor): The modified images with shape (B, C, H, W).
    
    Returns:
        torch.Tensor: The PSNR for each image in the batch.
    """
    mse = F.mse_loss(modified, original, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


def calc_ssim(original, modified, window_size=11, C1=0.01**2, C2=0.03**2):
    """Calculate the SSIM between two batches of images.

    Args:
        original (torch.Tensor): The original images with shape (B, C, H, W).
        modified (torch.Tensor): The modified images with shape (B, C, H, W).
    
    Returns:
        torch.Tensor: The SSIM for each image in the batch.
    """
    # Mean and variance for both images
    mu1 = F.avg_pool2d(original, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(modified, window_size, stride=1, padding=window_size//2)
    sigma1_sq = F.avg_pool2d(original * original, window_size, stride=1, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(modified * modified, window_size, stride=1, padding=window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(original * modified, window_size, stride=1, padding=window_size//2) - mu1 * mu2

    # SSIM formula
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean(dim=(1, 2, 3))  # Average SSIM over spatial dimensions
    return ssim


def calc_rmse(original, modified):
    """Calculate the RMSE between two batches of images.

    Args:
        original (torch.Tensor): The original images with shape (B, C, H, W).
        modified (torch.Tensor): The modified images with shape (B, C, H, W).

    Returns:
        torch.Tensor: The RMSE for each image in the batch.
    """
    mse = F.mse_loss(modified, original, reduction='none').mean(dim=(1, 2, 3))
    rmse = torch.sqrt(mse)
    return rmse


def calc_rmse_w(original, modified, mask):
    """Calculate the RMSE between two images in the watermarked region only.

    Args:
        original (torch.Tensor): The original images with shape (B, C, H, W).
        modified (torch.Tensor): The modified images with shape (B, C, H, W).
        mask (torch.Tensor): The mask of the watermarked region with shape (B, 1, H, W).
    
    Returns:
        torch.Tensor: The RMSE for each image in the watermarked region.
    """
    diff = (modified - original) ** 2
    masked_diff = diff * mask
    mse_w = masked_diff.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    rmse_w = torch.sqrt(mse_w)
    return rmse_w
