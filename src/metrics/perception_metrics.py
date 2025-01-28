import torch
import numpy as np
from skimage import img_as_ubyte, color
from imquality import brisque
# from skimage.metrics import niqe

# from nima.inference import predict_image_aesthetic_quality
# from nima.model import load_model

# Load the NIMA model globally for batch processing
# nima_model = load_model('mobilenet')

# Helper to convert a PyTorch tensor to a numpy image
def tensor_to_numpy(batch_tensor):
    """
    Converts a batch of PyTorch tensors to numpy arrays.
    :param batch_tensor: PyTorch tensor of shape (B, C, H, W) in range [0, 1]
    :return: List of numpy arrays [(H, W, C), ...]
    """
    batch_tensor = batch_tensor.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
    return [(img.cpu().numpy() * 255).astype(np.uint8) for img in batch_tensor]

def compute_batch_niqe(batch_tensor):
    """
    Compute NIQE for a batch of images.
    :param batch_tensor: PyTorch tensor of shape (B, C, H, W) in range [0, 1]
    :return: List of NIQE scores
    """
    batch_tensor = batch_tensor.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
    batch_images = batch_tensor.cpu().numpy()  # Convert to NumPy array
    niqe_scores = []
    
    for img in batch_images:
        # If image is RGB, convert to grayscale
        if img.shape[-1] == 3:
            img = color.rgb2gray(img)
        # Convert image to uint8
        img = img_as_ubyte(img)
        # Compute NIQE for the image
        score = niqe(img)
        niqe_scores.append(score)
    
    return niqe_scores


### 2. BRISQUE for Batch
def compute_batch_brisque(batch_tensor):
    """
    Compute BRISQUE for a batch of images.
    :param batch_tensor: PyTorch tensor of shape (B, C, H, W) in range [0, 1]
    :return: List of BRISQUE scores
    """
    images = tensor_to_numpy(batch_tensor)
    brisque_scores = [brisque.score(img) for img in images]
    return brisque_scores

# ### 3. MA Score using NIMA for Batch
# def compute_batch_ma_score(batch_tensor):
#     """
#     Compute Ma scores using the NIMA model for a batch of images.
#     :param batch_tensor: PyTorch tensor of shape (B, C, H, W) in range [0, 1]
#     :return: List of Ma scores
#     """
#     images = tensor_to_numpy(batch_tensor)
#     ma_scores = [predict_image_aesthetic_quality(img, model=nima_model)['mean_score'] for img in images]
#     return ma_scores

# ### 4. PI for Batch
# def compute_batch_pi(batch_tensor):
#     """
#     Compute Perceptual Index (PI) for a batch of images.
#     :param batch_tensor: PyTorch tensor of shape (B, C, H, W) in range [0, 1]
#     :return: List of PI scores
#     """
#     niqe_scores = compute_batch_niqe(batch_tensor)
#     ma_scores = compute_batch_ma_score(batch_tensor)
#     pi_scores = [0.5 * (niqe + (10 - ma)) for niqe, ma in zip(niqe_scores, ma_scores)]
#     return pi_scores
