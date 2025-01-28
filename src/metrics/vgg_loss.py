import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pytorch_ssim
from torchvision.models import vgg16

class CustomLoss(nn.Module):
    def __init__(self, perceptual_weight=0.2, alpha=0.4, beta=0.3, omega=0.1):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        
        # Load a pretrained VGG model for perceptual loss
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, xhat, target):
        xhat_features = self.vgg(xhat)
        target_features = self.vgg(target)
        return F.mse_loss(xhat_features, target_features)

    def edge_loss(self, predicted, target):
        # Define the Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)

        # Expand the Sobel kernels to match the number of input channels
        num_channels = predicted.shape[1]
        sobel_x = sobel_x.expand(num_channels, 1, 3, 3).to(predicted.device)
        sobel_y = sobel_y.expand(num_channels, 1, 3, 3).to(predicted.device)

        # Apply the Sobel filters to compute gradients
        edge_pred_x = F.conv2d(predicted, sobel_x, groups=num_channels, padding=1)
        edge_pred_y = F.conv2d(predicted, sobel_y, groups=num_channels, padding=1)
        edge_pred = torch.sqrt(edge_pred_x**2 + edge_pred_y**2)

        edge_target_x = F.conv2d(target, sobel_x, groups=num_channels, padding=1)
        edge_target_y = F.conv2d(target, sobel_y, groups=num_channels, padding=1)
        edge_target = torch.sqrt(edge_target_x**2 + edge_target_y**2)

        # Compute the L1 loss between predicted and target edges
        return F.l1_loss(edge_pred, edge_target)



    def forward(self, xhat, target):
        l1_loss = F.l1_loss(xhat, target)
        
        # Masked SSIM Loss
        ssim_loss = 1 - pytorch_ssim.ssim(xhat, target)
        
        # Perceptual Loss
        perceptual_loss = self.perceptual_loss(xhat, target)

        edge_loss = self.edge_loss(xhat, target)
        
        # Combine losses
        total_loss = (self.alpha * l1_loss +
                      self.beta * ssim_loss +
                      self.perceptual_weight * perceptual_loss)
        return total_loss




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from . import pytorch_ssim
# from torchvision.models import vgg16

# class CustomLoss(nn.Module):
#     def __init__(self, perceptual_weight=0.2, alpha=0.4, beta=0.4):
#         super().__init__()
#         self.perceptual_weight = perceptual_weight
#         self.alpha = alpha
#         self.beta = beta
        
#         # Load a pretrained VGG model for perceptual loss
#         self.vgg = vgg16(pretrained=True).features[:16].eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def perceptual_loss(self, xhat, target):
#         xhat_features = self.vgg(xhat)
#         target_features = self.vgg(target)
#         return F.mse_loss(xhat_features, target_features)

#     def forward(self, xhat, target, mask):
#         # Masked MSE Loss
#         mask_area = mask > 0.5  # Binary mask
#         mask_area = mask_area.expand_as(xhat)
#         masked_mse_loss = F.mse_loss(xhat[mask_area], target[mask_area])
        
#         # Masked SSIM Loss
#         ssim_loss = 1 - pytorch_ssim.ssim(xhat * mask, target * mask)
        
#         # Perceptual Loss
#         perceptual_loss = self.perceptual_loss(xhat, target)
        
#         # Combine losses
#         total_loss = (self.alpha * masked_mse_loss +
#                       self.beta * ssim_loss +
#                       self.perceptual_weight * perceptual_loss)
#         return total_loss





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from . import pytorch_ssim
# from torchvision.models import vgg16

# class CustomLoss(nn.Module):
#     def __init__(self, perceptual_weight=0.2, alpha=0.4, beta=0.4):
#         super().__init__()
#         self.perceptual_weight = perceptual_weight
#         self.alpha = alpha
#         self.beta = beta
        
#         # Load a pretrained VGG model for perceptual loss
#         self.vgg = vgg16(pretrained=True).features[:16].eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def perceptual_loss(self, xhat, target):
#         xhat_features = self.vgg(xhat)
#         target_features = self.vgg(target)
#         return F.mse_loss(xhat_features, target_features)

#     def forward(self, xhat, target, mask):
#         # Masked MSE Loss
#         mask_area = mask > 0.5  # Binary mask
#         mask_area = mask_area.expand_as(xhat)
#         masked_mse_loss = F.mse_loss(xhat[mask_area], target[mask_area])
        
#         # Masked SSIM Loss
#         ssim_loss = 1 - pytorch_ssim.ssim(xhat * mask, target * mask)
        
#         # Perceptual Loss
#         perceptual_loss = self.perceptual_loss(xhat, target)
        
#         # Combine losses
#         total_loss = (self.alpha * masked_mse_loss +
#                       self.beta * ssim_loss +
#                       self.perceptual_weight * perceptual_loss)
#         return total_loss
