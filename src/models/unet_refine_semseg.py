import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class UNetRefineSemseg(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v2", c_in: int = 4):
        """
        U-Net model for mask refinement.

        Args:
            backbone: the type of backbone for the U-net model
            c_in: number of color channels on the input image
        """    
        super(UNetRefineSemseg, self).__init__()
        model = smp.Unet(
            encoder_name=backbone,       # backbone
            encoder_weights="imagenet",  # pre-trained weights
            in_channels=c_in,            # model input channels
            classes=2
        )
        
        #
        # cut off segmentation head
        #
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                model.decoder.blocks[-1].conv2[0].out_channels,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )



    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)     
        features = self.encoder(x)
        x = self.decoder(*features)
        out = self.final_conv(x)

        return out