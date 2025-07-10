# encoder of the VAE we will use for stable diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_attentionblock, VAE_residualblock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super(VAE_Encoder, self).__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # B, 128, H, W
            VAE_residualblock(128, 128),  # B, 128, H, W
            VAE_residualblock(128, 128),  # B, 128, H, W
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1),  # B, 128, H/2, W/2
            VAE_residualblock(128, 256),  # B, 256, H/2, W/2
            VAE_residualblock(256, 256),  # B, 256, H/2, W/2
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # B, 256, H/4, W/4
            VAE_residualblock(256, 512),  # B, 512, H/4, W/4
            VAE_residualblock(512, 512),  # B, 512, H/4, W/4
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_attentionblock(512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            nn.GroupNorm(32, 512),  # B, 512, H/8, W/8
            nn.SiLU(),  # B, 512, H/8, W/8
            nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1),  # B, 8, H/8, W/8
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # B, 8, H/8, W/8
        ) 
        
    def forward(self, x, noise):
        # x: B, 3, H, W
        # noise: B, 4, H/8, W/8  
        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # B, 8, H/8, W/8 -> 2 * B, 4, H/8, W/8
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()   
        stdev = variance.sqrt()
        # Transform N(0, 1) -> N(mean, stdev) 
        # reparameterization trick
        x = mean + stdev * noise
        # Rescale the output to match the range of the input image
        x *= 0.18215
        return x