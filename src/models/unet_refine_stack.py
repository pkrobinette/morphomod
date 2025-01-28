import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class UNetRefineStack(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v2", img_channels: int = 3, mask_channels: int = 1):
        """
        U-Net model for refinement.

        Args:
            backbone: the type of backbone for the U-net model
            img_channels: number of channels in the input image
            mask_channels: number of channels in the input mask
        """    
        super(UNetRefineStack, self).__init__()
        
        # Total input channels: image channels + mask channels
        c_in = img_channels + mask_channels
        
        model = smp.Unet(
            encoder_name=backbone,       # backbone
            encoder_weights="imagenet",  # pre-trained weights
            in_channels=c_in,            # model input channels
            classes=img_channels          # output channels same as image channels
        )
        
        # Cut off segmentation head
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                model.decoder.blocks[-1].conv2[0].out_channels,
                img_channels,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, image, mask):
        """
        Forward pass with both image and mask as input.

        Args:
            image: Tensor of shape (B, img_channels, H, W)
            mask: Tensor of shape (B, mask_channels, H, W)
        
        Returns:
            out: Refined image of shape (B, img_channels, H, W)
        """
        # Concatenate image and mask along the channel dimension
        x = torch.cat([image, mask], dim=1)
        
        # Pass through the model
        features = self.encoder(x)
        x = self.decoder(*features)
        out = self.final_conv(x)

        return out
