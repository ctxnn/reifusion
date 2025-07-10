# decoder of the VAE we will use for stable diffusion

import torch
import torch.nn as nn   
import torch.nn.functional as F 
from attention import SelfAttention

class VAE_attentionblock(nn.Module):    
    def __init__(self, in_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(in_channels)  # B, in_channels, H, W
                
    def forward(self, x):
        residue = x  # B, in_channels, H, W
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w) 
        x = x.transpose(1, 2)  # B, H*W, in_channels
        x = self.attention(x)  # B, H*W, in_channels
        x = x.transpose(1, 2)  # B, in_channels, H
        x = x.view(n, c, h, w)  # B, in_channels, H, W
        return x + residue  # B, in_channels, H, W

 
class VAE_residualblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0) # B, out_channels, H, W
        
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0) # B, out_channels, H, W 
        
        if in_channels != out_channels:
                # if the channel counts don't match, we can't directly add the two tensors.
                # to fix this, we use a 1x1 convolution. This special type of convolution
                # acts as a "projector" or "adapter." It changes the number of channels
                # from `in_channels` to `out_channels` without changing the height or width
                # of the feature map.
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
                # if the input and output channel counts are already the same, no change is needed.
                # `nn.Identity()` is a placeholder layer that simply passes the input through
                # without any modification. It's like a direct wire.
            self.skip = nn.Identity()
            
    def forward(self, x):
        # x: B, in_channels, H, W
        # apply the first group normalization and convolution
        residue = x 
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        # apply the second group normalization and convolution
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.skip(residue)
          
