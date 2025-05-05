import torch
import numpy as np
from scipy.ndimage import binary_dilation
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from torchvision import transforms as T
from typing import Union
from huggingface_hub import login
from diffusers import FluxFillPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

inpaint_mods = {
    'SD2': "stabilityai/stable-diffusion-2-inpainting",
    'SDXL': "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    'LaMa': "tbd",
    'FLUX': "black-forest-labs/FLUX.1-Fill-dev"
}

# TOKEN=<FILL IN>


class MorphoModel(nn.Module):
    def __init__(self, mask_model, refine_model, inpaint: str = 'SD2'):
        """
        MorphoModel. --> mask of watermark + refine + dilate.

        Args:
            mask_model: model to get mask of watermark.
            refine_model: model used to refine the mask.
        """    
        super(MorphoModel, self).__init__()
        
        self.device = DEVICE
        self.mask_model = mask_model.to(DEVICE)
        self.refine_model = refine_model.to(DEVICE)

        if inpaint == None: inpaint = 'SD2'
        assert inpaint in inpaint_mods, "Inpaint model not available. [SD2, SDXL, LaMa] Available."
        if inpaint == 'SD2':
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16
            ).to(DEVICE)
        elif inpaint == 'SDXL':
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(DEVICE)
        elif inpaint == "FLUX": 
            login(token=TOKEN)
            self.pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev", 
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
        else:
            raise ValueError("Model not implemented.")
        
        self.transform = T.ToTensor()


    def forward(self, x, dilate: int = 1, prompt: str = "Remove."):
        masks = self.get_mask(x, dilate).to(self.device)
        xhat = self.pipe(
            prompt=[prompt]*x.shape[0], 
            image=x, 
            mask_image=masks).images
        # convert to tensors
        xhat_t = [self.transform(im.resize(x[0].shape[1:])) for im in xhat]
        xhat_out = torch.stack([t.detach() for t in xhat_t]).to(self.device)
        imfinal = (xhat_out * masks) + ((1-masks)*x)

        return imfinal, masks

    def forward_w_gt(
        self, 
        x, 
        masks, 
        dilate: int = 1, 
        prompt: str = "Remove.", 
        fill: Union[str, None] = None,
        num_steps: int = 50
    ):
        masks = self.dilate_masks(masks, dilate).to(self.device)
        if fill == "background":
            x = fill_with_background(x, masks)
        elif fill in ["black", "white", "gray"]:
            x = fill_with_color(x, masks, fill)
        xhat = self.pipe(
            prompt=[prompt]*x.shape[0], 
            image=x, 
            mask_image=masks,
            num_inference_steps=num_steps
        ).images
        # convert to tensors
        xhat_t = [self.transform(im.resize(x[0].shape[1:])) for im in xhat]
        raw_out = torch.stack([t.detach() for t in xhat_t]).to(self.device)
        imfinal = (raw_out * masks) + ((1-masks)*x)

        return imfinal, masks, raw_out, x
        

    def get_mask(self, x, dilate: int = 1):
        """
        Forward pass with the mask

        Args:
            x: Tensor of shape (B, img_channels, H, W)
            dilate: How much to dilate gen mask
        
        Returns:
            out: Refined mask (B, mask_channels, H, W)
        """
        #
        # Gen mask
        #
        gen_masks = self.mask_model(x)
        gen_masks = torch.where(
            gen_masks > 0.5,
            torch.ones_like(gen_masks),
            torch.zeros_like(gen_masks)
        ).to(self.device)
        #
        # Refine Mask
        #
        refine_masks = self.refine_model(x, gen_masks)
        refine_masks = torch.where(
            refine_masks > 0.5,
            torch.ones_like(refine_masks),
            torch.zeros_like(refine_masks)
        ).to(self.device)
        #
        # Dilate
        #
        d_masks = self.dilate_masks(refine_masks, dilate)

        return d_masks

    def dilate_masks(self, batch_masks: torch.Tensor, dilation_iterations: int = 1):
        """
        Apply morphological dilation to a batch of binary segmentation masks.
    
        Args:
            batch_masks: A batch of binary masks with shape (batch_size, height, width).
                     Each mask should be binary (0 or 1).
            dilation_iterations : Number of dilation iterations to apply. 
                    Higher values increase the dilation.
    
        Returns:
            A batch of dilated masks with the same shape as the input.
        """
        # if no dilate, do not dilate
        if dilation_iterations == 0:
            return batch_masks
        #
        # Check type
        # 
        if not isinstance(batch_masks, torch.Tensor):
            raise ValueError("batch_masks must be a torch.Tensor.")
        #
        # Make sure the right dims
        # 
        if batch_masks.ndim == 4:
            batch_masks = batch_masks.squeeze(1)
        if batch_masks.ndim != 3:
            raise ValueError("batch_masks must have 3 dimensions: (batch_size, height, width).")
        #
        # Dilate
        #
        batch_masks_np = batch_masks.cpu().numpy()
    
        # apply dilation to each mask in the batch
        dilated_masks_np = np.array([
            binary_dilation(mask, iterations=dilation_iterations).astype(np.uint8)
            for mask in batch_masks_np
        ])
    
        #  back to torch.Tensor
        dilated_masks = torch.tensor(dilated_masks_np, dtype=torch.uint8, device=batch_masks.device)
    
        return dilated_masks.unsqueeze(1)


def fill_with_background(image: torch.Tensor, mask: torch.Tensor):
    """
    Fills the background of the image with the most prevalent background pixel values.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        mask (torch.Tensor): Binary mask tensor of shape (B, 1, H, W) where 1 represents the object and 0 the background.

    Returns:
        torch.Tensor: Image with the background filled.
    """
    filled_images = []
    batch_size, _, height, width = image.shape

    for b in range(batch_size):
        img = image[b].cpu().numpy()  # Convert to numpy array for processing
        msk = mask[b, 0].cpu().numpy()  # Extract single channel mask

        # Get pixel values where the mask is 0 (background)
        background_pixels = img[:, msk == 0]

        # Calculate the most prevalent pixel value for each channel
        prevalent_pixel_value = np.array([np.median(background_pixels[c]) for c in range(img.shape[0])])

        # Create the filled background
        background_filled = img.copy()
        for c in range(img.shape[0]):
            background_filled[c, msk == 1] = prevalent_pixel_value[c]

        filled_images.append(torch.tensor(background_filled, device=image.device))

    return torch.stack(filled_images)
    

def fill_with_color(image: torch.Tensor, mask: torch.Tensor, color: str):
    """
    Fills the masked region of the image with black (pixel value 0).

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        mask (torch.Tensor): Binary mask tensor of shape (B, 1, H, W) where 1 represents the region to be filled with black.

    Returns:
        torch.Tensor: Image with the masked region filled with black.
    """
    # Ensure mask is broadcastable to the image shape
    mask_expanded = mask.expand_as(image)  # (B, C, H, W)

    # Set the regions where mask == 1 to black (pixel value 0)
    filled_image = image.clone()  # Clone the input image to avoid in-place modifications
    if color == "black":
        filled_image[mask_expanded == 1] = 0
    elif color == "white":
        filled_image[mask_expanded == 1] = 1
    elif color == "gray":
        filled_image[mask_expanded == 1] = 0.5
    else:
        raise ValueError("Incorrect color value. Must be [black, white, or gray]")
        
    return filled_image


def dilate(batch_masks, dilation_iterations=1):
    """
    Apply morphological dilation to a batch of binary segmentation masks.

    Parameters:
    batch_masks (torch.Tensor): A batch of binary masks with shape (batch_size, height, width).
                                Each mask should be binary (0 or 1).
    dilation_iterations (int): Number of dilation iterations to apply. Higher values increase the dilation.

    Returns:
    torch.Tensor: A batch of dilated masks with the same shape as the input.
    """
    if dilation_iterations == 0:
        return batch_masks
        
    if not isinstance(batch_masks, torch.Tensor):
        raise ValueError("batch_masks must be a torch.Tensor.")

    if batch_masks.ndim == 4:
        batch_masks = batch_masks.squeeze(1)

    if batch_masks.ndim != 3:
        raise ValueError("batch_masks must have 3 dimensions: (batch_size, height, width).")

    # convert to numpy for dilation
    batch_masks_np = batch_masks.cpu().numpy()

    # apply dilation to each mask in the batch
    dilated_masks_np = np.array([
        binary_dilation(mask, iterations=dilation_iterations).astype(np.uint8)
        for mask in batch_masks_np
    ])

    #  back to torch.Tensor
    dilated_masks = torch.tensor(dilated_masks_np, dtype=torch.uint8, device=batch_masks.device)

    return dilated_masks.unsqueeze(1)