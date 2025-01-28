import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class UNetSemseg(nn.Module):
    def __init__(self, classes=2, backbone="mobilenet_v2", c_in=3):
        """
        U-Net model with a semantic segmentation head.

        :param classes: number of classification classes
        :type classes: int
        :param backbone: the type of backbone for the U-net model
        :type backbone: str
        :param c_in: number of color channels on the input image
        :type c_in: int
        """    
        super(UNetSemseg, self).__init__()
        model = smp.Unet(
            encoder_name=backbone,       # backbone
            encoder_weights="imagenet",  # pre-trained weights
            in_channels=c_in,            # model input channels
            classes=classes
        )
        #
        # cut off segmentation head
        #
        self.encoder = model.encoder
        self.decoder = model.decoder
        #
        # Add semseg layer
        #
        self.segmentation_head = model.segmentation_head


    def forward(self, x):     
        x = self.encoder(x)
        x = self.decoder(*x) # have to star here

        semseg = self.segmentation_head(x)

        return semseg
    

    def generate(self, x):       
        semseg = self.forward(x)
        
        return torch.argmax(semseg, dim=1)
