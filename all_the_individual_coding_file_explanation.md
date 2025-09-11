# Deep Code Reference - Stable Diffusion Implementation

This document provides line-by-line analysis of every component in the Stable Diffusion implementation.

## Table of Contents
1. [Attention Mechanisms (`attention.py`)](#attention-mechanisms)
2. [VAE Decoder (`decoder.py`)](#vae-decoder)
3. [VAE Encoder (`encoder.py`)](#vae-encoder)
4. [CLIP Text Encoder (`clip.py`)](#clip-text-encoder)
5. [Diffusion U-Net (`diffusion.py`)](#diffusion-u-net)

---

## Attention Mechanisms (`attention.py`)

### SelfAttention Class

```python
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
```
**Line Analysis:**
- Standard PyTorch imports for neural network operations
- `math` needed for square root calculations in attention scaling

```python
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
```
**Parameters:**
- `n_heads`: Number of attention heads for multi-head attention
- `d_embed`: Embedding dimension (must be divisible by n_heads)
- `in_proj_bias`: Whether to use bias in input projection (Q, K, V)
- `out_proj_bias`: Whether to use bias in output projection

```python
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)  # B, L, 3 * d_embed
```
**Purpose:** Single linear layer that projects input to Q, K, V simultaneously
- Input: (B, L, d_embed) 
- Output: (B, L, 3 * d_embed) - concatenated [Q, K, V]
- More efficient than 3 separate linear layers

```python
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_ehead = d_embed // n_heads
```
**Note:** There's a typo - should be `self.d_head` not `self.d_ehead`
- `out_proj`: Final projection after attention computation
- `d_head`: Dimension per attention head (d_embed / n_heads)

```python
    def forward(self, x, causal_mask=False):
        # x: B, L, d_embed
        batch_size, seq_len, d_embed = x.shape
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)
```
**Shape tracking:**
- Input `x`: (batch_size, sequence_length, embedding_dimension)
- `interm_shape`: Target shape for reshaping Q, K, V for multi-head attention

```python
        # Project input to query, key, value
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # B, L, D -> B, L, dim*3 -> 3*(B, L, dim)
```
**Operation breakdown:**
1. `self.in_proj(x)`: (B, L, d_embed) → (B, L, 3*d_embed)
2. `.chunk(3, dim=-1)`: Split last dimension into 3 equal parts
3. Result: 3 tensors each of shape (B, L, d_embed)

```python
        q = q.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head
        k = k.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head
        v = v.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head
```
**Multi-head reshaping:**
1. `.view(interm_shape)`: (B, L, d_embed) → (B, L, n_heads, d_head)
2. `.transpose(1, 2)`: (B, L, H, d_head) → (B, H, L, d_head)
3. Final shape allows parallel computation across heads

```python
        weight = q @ k.transpose(-1, -2)  # B, H, L, d_head @ B, H, d_head, L -> B, H, L, L
```
**Attention score computation:**
- `k.transpose(-1, -2)`: (B, H, L, d_head) → (B, H, d_head, L)
- `q @ k.T`: Matrix multiplication gives attention scores
- Result: (B, H, L, L) - each position attends to every other position

```python
        if causal_mask: 
            #upper traingle is made up of 1 then fill up with -infinity(only upper traingle though)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            weight.masked_fill_(mask, float('-inf'))  # Set upper triangle to -inf  
```
**Causal masking (for autoregressive generation):**
- `torch.triu(..., diagonal=1)`: Upper triangular matrix (excluding diagonal)
- Prevents positions from attending to future positions
- `-inf` values become 0 after softmax

```python
        weight = weight / (self.d_head ** 0.5)  # Scale by sqrt(d_head) 
```
**Scaled dot-product attention:**
- Scaling prevents softmax saturation with large d_head
- Mathematical derivation: prevents gradients from vanishing

```python
        weight = weight.softmax(dim=-1)  # B, H, L, L -> B, H, L, L
```
**Attention probability computation:**
- Softmax over last dimension (attended positions)
- Each row sums to 1, representing probability distribution

```python
        output = weight @ v  # B, H, L, L @ B, H, L, d_head -> B, H, L, d_head
```
**Weighted value aggregation:**
- Multiply attention weights by values
- Result: Each position gets weighted sum of all values

```python
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_embed)  # B, H, L, d_head -> B, L, H, d_head -> B, L, D
```
**Reshape back to original format:**
1. `.transpose(1, 2)`: (B, H, L, d_head) → (B, L, H, d_head)
2. `.contiguous()`: Ensures memory layout for efficient view
3. `.view(...)`: (B, L, H, d_head) → (B, L, d_embed)

```python
        output = self.out_proj(output)  # B, L, D
        return output
```
**Final projection:**
- Combines information from all heads
- Allows model to learn optimal head combination

### CrossAttention Class

```python
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
```
**Key difference from SelfAttention:**
- `d_cross`: Dimension of the cross-attention input (e.g., text embeddings)
- Allows attention between different modalities (image ↔ text)

```python
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
```
**Separate projections:**
- `q_proj`: Projects queries from main input (d_embed → d_embed)
- `k_proj`, `v_proj`: Projects keys/values from cross input (d_cross → d_embed)
- All projected to same dimension for compatibility

```python
    def forward(self, x, y):
        # x (Query): B, L_q, D_embed
        # y (Key, Value): B, L_kv, D_cross
```
**Input specification:**
- `x`: Main sequence (e.g., image features)
- `y`: Cross sequence (e.g., text embeddings)
- Different sequence lengths allowed (L_q ≠ L_kv)

```python
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        # B, L_kv, H, d_head
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
```
**Shape preparation:**
- Store original shape for final reshape
- `interim_shape` with `-1` for flexible key/value length

```python
        # Project query, key, value
        q = self.q_proj(x) # B, L_q, D_embed
        k = self.k_proj(y) # B, L_kv, D_embed
        v = self.v_proj(y) # B, L_kv, D_embed
```
**Cross-modal projections:**
- Query from main input (image features)
- Key and Value from cross input (text features)
- All projected to common embedding space

```python
        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2) # B, H, L_q, d_head
        k = k.view(interim_shape).transpose(1, 2) # B, H, L_kv, d_head
        v = v.view(interim_shape).transpose(1, 2) # B, H, L_kv, d_head
```
**Multi-head reshaping:**
- Each tensor reshaped for parallel head computation
- Query and Key/Value can have different sequence lengths

```python
        # Calculate attention scores
        weight = q @ k.transpose(-1, -2) # B, H, L_q, d_head @ B, H, d_head, L_kv -> B, H, L_q, L_kv
```
**Cross-attention scores:**
- Each query position attends to all key positions
- Shape: (B, H, L_q, L_kv) - potentially rectangular matrix

```python
        # Scale
        weight /= math.sqrt(self.d_head)
        
        # Softmax
        weight = F.softmax(weight, dim=-1) # B, H, L_q, L_kv
```
**Standard attention operations:**
- Scale by sqrt(d_head) for stability
- Softmax over key positions (last dimension)

```python
        # Apply attention to value
        output = weight @ v # B, H, L_q, L_kv @ B, H, L_kv, d_head -> B, H, L_q, d_head
```
**Weighted aggregation:**
- Each query gets weighted combination of all values
- Output length matches query length

```python
        # Reshape and transpose back
        output = output.transpose(1, 2).contiguous() # B, L_q, H, d_head
        output = output.view(input_shape) # B, L_q, D_embed
        
        # Final projection
        output = self.out_proj(output) # B, L_q, D_embed
        return output
```
**Final processing:**
- Reshape back to original query format
- Output projection for learned combination

---

## VAE Decoder (`decoder.py`)

### Imports and Dependencies

```python
import torch
import torch.nn as nn   
import torch.nn.functional as F 
from attention import SelfAttention
```
**Dependencies:**
- Standard PyTorch modules
- Custom SelfAttention for spatial attention in high-resolution features

### VAE_attentionblock Class

```python
class VAE_attentionblock(nn.Module):    
    def __init__(self, in_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(in_channels)  # B, in_channels, H, W
```
**Architecture choice:**
- `GroupNorm(32, ...)`: Normalizes groups of 32 channels
- `SelfAttention(in_channels)`: **ERROR** - should be `SelfAttention(1, in_channels)` (n_heads, d_embed)
- Used for spatial attention in feature maps

```python
    def forward(self, x):
        residue = x  # B, in_channels, H, W
        x = self.groupnorm(x)
```
**Residual connection setup:**
- Store input for skip connection
- Normalize before attention (standard Transformer pattern)

```python
        n, c, h, w = x.shape
        x = x.view(n, c, h * w) 
        x = x.transpose(1, 2)  # B, H*W, in_channels (to make attention work across spatial positions rather than channels)
```
**Spatial flattening for attention:**
1. Extract dimensions: batch, channels, height, width
2. `.view(n, c, h * w)`: Flatten spatial dimensions → (B, C, H*W)
3. `.transpose(1, 2)`: → (B, H*W, C)
4. **Purpose:** Treat each spatial position as a sequence element

```python
        x = self.attention(x)  # B, H*W, in_channels
        x = x.transpose(1, 2)  # B, in_channels, H
        x = x.view(n, c, h, w)  # B, in_channels, H, W
        return x + residue  # B, in_channels, H, W
```
**Attention and reshape back:**
1. Apply self-attention across spatial positions
2. Transpose back: (B, H*W, C) → (B, C, H*W)
3. Reshape to original: (B, C, H, W)
4. Add residual connection

### VAE_residualblock Class

```python
class VAE_residualblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0) # B, out_channels, H, W
```
**First convolution block:**
- `GroupNorm(32, ...)`: Group normalization with 32 channels per group
- `Conv2d(..., padding=0)`: **Important:** No padding - will reduce spatial size
- kernel_size=3 with padding=0 reduces H,W by 2 each

```python
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0) # B, out_channels, H, W 
```
**Second convolution block:**
- Same pattern: GroupNorm + Conv2d
- Again no padding - further spatial reduction

```python
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
```
**Skip connection handling:**
- **Channel mismatch:** Use 1×1 conv to project channels
- **Channel match:** Identity mapping (no operation)
- **Issue:** Skip connection won't work due to spatial size mismatch from padding=0

```python
    def forward(self, x):
        # x: B, in_channels, H, W
        # apply the first group normalization and convolution
        residue = x 
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
```
**First block forward:**
- Store residual before any processing
- SiLU activation (Sigmoid Linear Unit): `x * sigmoid(x)`
- More stable than ReLU for gradients

```python
        # apply the second group normalization and convolution
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.skip(residue)
```
**Second block and residual:**
- Same pattern: GroupNorm → SiLU → Conv
- **Problem:** `x` shape is (H-4, W-4) but `self.skip(residue)` is (H, W)
- **Should use:** Proper padding or adjust skip connection

### VAE_Decoder Class

```python
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        # Input: (B, 4, H/8, W/8)
        super().__init__(
```
**Architecture pattern:**
- Inherits from `nn.Sequential` for simple forward pass
- Input is latent representation from diffusion process

```python
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # (B, 4, H/8, W/8)
```
**Identity-like projection:**
- 1×1 conv with same input/output channels
- Could be for parameter initialization or slight transformation

```python
            nn.Conv2d(4, 512, kernel_size=3, padding=1), # (B, 512, H/8, W/8)
```
**Channel expansion:**
- 4 → 512 channels (128× increase)
- padding=1 maintains spatial dimensions
- Provides rich feature representation for upsampling

```python
            VAE_residualblock(512, 512), # (B, 512, H/8, W/8)
            VAE_attentionblock(512), # (B, 512, H/8, W/8)
            VAE_residualblock(512, 512), # (B, 512, H/8, W/8)
            VAE_residualblock(512, 512), # (B, 512, H/8, W/8)
            VAE_residualblock(512, 512), # (B, 512, H/8, W/8)
            VAE_residualblock(512, 512), # (B, 512, H/8, W/8)
```
**Deep feature processing at H/8 resolution:**
- Multiple residual blocks for complex feature learning
- One attention block for spatial relationships
- All maintain 512 channels and H/8×W/8 resolution

```python
            nn.Upsample(scale_factor=2), # (B, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (B, 512, H/4, W/4)
```
**First upsampling stage:**
- `Upsample(scale_factor=2)`: Nearest neighbor upsampling (2× spatial increase)
- Conv2d: Refines upsampled features, maintains 512 channels

```python
            VAE_residualblock(512, 512), # (B, 512, H/4, W/4)
            VAE_residualblock(512, 512), # (B, 512, H/4, W/4)
            VAE_residualblock(512, 512), # (B, 512, H/4, W/4)
```
**Processing at H/4 resolution:**
- 3 residual blocks for feature refinement
- Maintains high channel count (512) for rich representations

```python
            # repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size)
            nn.Upsample(scale_factor=2), # (B, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (B, 512, H/2, W/2)
            VAE_residualblock(512, 256), # (B, 256, H/2, W/2)
            VAE_residualblock(256, 256), # (B, 256, H/2, W/2)
            VAE_residualblock(256, 256), # (B, 256, H/2, W/2)
```
**Second upsampling stage:**
- Another 2× spatial upsampling
- Begin channel reduction: 512 → 256
- 3 residual blocks at H/2 resolution

```python
            nn.Upsample(scale_factor=2), # (B, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (B, 256, H, W)
            VAE_residualblock(256, 128), # (B, 128, H, W)
            VAE_residualblock(128, 128), # (B, 128, H, W)
            VAE_residualblock(128, 128), # (B, 128, H, W)
```
**Final upsampling stage:**
- Third 2× upsampling reaches full resolution
- Channel reduction: 256 → 128
- 3 residual blocks at full resolution

```python
            nn.GroupNorm(32, 128), # (B, 128, H, W)
            nn.SiLU(), # (B, 128, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # (B, 3, H, W)
        )
```
**Final output layers:**
- GroupNorm + SiLU: Final normalization and activation
- Conv2d(128, 3): Reduce to RGB channels
- Output: (B, 3, H, W) - final RGB image

```python
    def forward(self, x):
        # x: B, 4, H/8, W/8 
        
        x /= 0.18215 
```
**Scaling reversal:**
- Reverse the scaling applied in encoder
- 0.18215 is standard VAE scaling factor for Stable Diffusion

```python
        for module in self: 
            x = module(x) 
            
        return x # B, 3, H, W
```
**Sequential processing:**
- Apply each layer in sequence
- Final output is RGB image

---

## VAE Encoder (`encoder.py`)

### Imports and Setup

```python
# encoder of the VAE we will use for stable diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_attentionblock, VAE_residualblock
```
**Dependencies:**
- Standard PyTorch imports
- Imports building blocks from decoder module
- Reuses same residual and attention components

### VAE_Encoder Class

```python
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super(VAE_Encoder, self).__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # B, 128, H, W
```
**Initial feature extraction:**
- Input: RGB image (3 channels)
- Output: 128 feature channels
- padding=1 maintains spatial dimensions

```python
            VAE_residualblock(128, 128),  # B, 128, H, W
            VAE_residualblock(128, 128),  # B, 128, H, W
```
**Initial processing blocks:**
- 2 residual blocks at full resolution
- Build rich feature representations before downsampling

```python
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1),  # B, 128, H/2, W/2
```
**First downsampling:**
- stride=2: Reduces spatial dimensions by half
- Maintains 128 channels
- padding=1 with stride=2 gives exact 2× reduction

```python
            VAE_residualblock(128, 256),  # B, 256, H/2, W/2
            VAE_residualblock(256, 256),  # B, 256, H/2, W/2
```
**Processing at H/2 resolution:**
- Channel expansion: 128 → 256
- 2 residual blocks for feature learning

```python
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # B, 256, H/4, W/4
```
**Second downsampling:**
- Another 2× spatial reduction
- Maintains 256 channels

```python
            VAE_residualblock(256, 512),  # B, 512, H/4, W/4
            VAE_residualblock(512, 512),  # B, 512, H/4, W/4
```
**Processing at H/4 resolution:**
- Channel expansion: 256 → 512
- 2 residual blocks

```python
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # B, 512, H/8, W/8
```
**Final downsampling:**
- Third 2× reduction reaches H/8×W/8
- Total 8× downsampling (2³)

```python
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
            VAE_attentionblock(512),  # B, 512, H/8, W/8
            VAE_residualblock(512, 512),  # B, 512, H/8, W/8
```
**Deep processing at latent resolution:**
- 4 residual blocks + 1 attention block
- Attention captures long-range spatial dependencies
- All maintain 512 channels

```python
            nn.GroupNorm(32, 512),  # B, 512, H/8, W/8
            nn.SiLU(),  # B, 512, H/8, W/8
```
**Pre-output normalization:**
- GroupNorm: Stabilizes features before final projection
- SiLU: Smooth activation

```python
            nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1),  # B, 8, H/8, W/8
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # B, 8, H/8, W/8
        ) 
```
**Output projection:**
- First conv: 512 → 8 channels
- Second conv: 8 → 8 (refinement)
- Output will be split into mean and log_variance (4 channels each)

### Forward Pass with Reparameterization

```python
    def forward(self, x, noise):
        # x: B, 3, H, W
        # noise: B, 4, H/8, W/8  
```
**Input specification:**
- `x`: RGB image
- `noise`: Random noise for reparameterization trick

```python
        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
```
**Asymmetric padding for downsampling:**
- **Purpose:** Ensures proper spatial dimensions with stride=2
- `F.pad(x, (0, 1, 0, 1))`: Pad right and bottom by 1 pixel
- **Reason:** With odd dimensions, symmetric padding doesn't work cleanly
- Reference to issue #8 suggests this was a discovered bug

```python
            x = module(x)
```
**Sequential processing:**
- Apply each module in the defined order
- Handle padding only for stride-2 convolutions

```python
        # B, 8, H/8, W/8 -> 2 * B, 4, H/8, W/8
        mean, log_variance = torch.chunk(x, 2, dim=1)
```
**Split output into mean and log-variance:**
- `torch.chunk(x, 2, dim=1)`: Split 8 channels into 2×4 channels
- `mean`: μ for Gaussian distribution
- `log_variance`: log(σ²) for numerical stability

```python
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        log_variance = torch.clamp(log_variance, -30, 20)
```
**Variance clamping:**
- Prevents extreme values that cause numerical instability
- `exp(-30) ≈ 1e-14`: Minimum variance (almost deterministic)
- `exp(20) ≈ 5e8`: Maximum variance (very noisy)

```python
        variance = log_variance.exp()   
        stdev = variance.sqrt()
```
**Convert to standard deviation:**
- `exp(log_variance)`: Get actual variance
- `sqrt(variance)`: Get standard deviation for sampling

```python
        # Transform N(0, 1) -> N(mean, stdev) 
        # reparameterization trick
        x = mean + stdev * noise
```
**Reparameterization trick:**
- Sample from N(μ, σ) using N(0,1) noise
- **Mathematical:** If Z ~ N(0,1), then μ + σZ ~ N(μ, σ²)
- **Purpose:** Makes sampling differentiable for backpropagation

```python
        # Rescale the output to match the range of the input image
        x *= 0.18215
        return x
```
**Final scaling:**
- 0.18215: Standard scaling factor for Stable Diffusion VAE
- Empirically determined for optimal training stability

---

## CLIP Text Encoder (`clip.py`)

### Imports

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention
```
**Dependencies:**
- Standard PyTorch modules
- Custom SelfAttention for text processing

### CLIPEmbedding Class

```python
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embd, n_tokens):
        
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
```
**Embedding layers:**
- `token_embedding`: Maps vocabulary indices to dense vectors
- `position_embedding`: Learned positional encodings (not sinusoidal)
- `n_vocab=49408`: CLIP vocabulary size
- `n_embd=768`: Embedding dimension
- `n_tokens=77`: Maximum sequence length

```python
    def forward(self, tokens):
        x = self.token_embedding(tokens)  # (B, L, D)
        x += self.position_embedding
        return x 
```
**Embedding forward pass:**
- Map token IDs to embeddings
- Add position embeddings (broadcasted across batch)
- **Note:** No dropout, different from some Transformer implementations

### CLIPLayer Class

```python
class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
```
**Standard Transformer layer components:**
- `layernorm1`: Pre-attention normalization
- `attention`: Multi-head self-attention
- `layernorm2`: Pre-FFN normalization
- `linear1/2`: Feed-forward network (4× expansion)

```python
    def forward(self, x):
        # B, L, D = x.shape

        residue = x
        x = self.layernorm1(x)  # layer normalization
        attn_output = self.attention(x, casual_mask=True)  # (B, L, D)
        x += residue  # residual connection
```
**Attention block:**
- **Pre-norm:** LayerNorm before attention (modern Transformer style)
- `casual_mask=True`: **Typo** - should be `causal_mask`
- **Causal masking:** Each token only sees previous tokens
- Residual connection around attention

```python
        residue = x
        x = self.layernorm2(x)  # layer normalization

        # Feed-forward network
        x = self.linear1(x)  # (B, L, 4D)
        x = x * torch.sigmoid(1.702 * x)  # quick gelu activation
        x = self.linear2(x)  # (B, L, D)
        x += residue  # residual connection
        return x
```
**Feed-forward block:**
- Pre-norm before FFN
- **Quick GELU:** `x * sigmoid(1.702 * x)` 
  - Approximates GELU: `x * Φ(x)` where Φ is standard normal CDF
  - 1.702 ≈ √(2/π) for mathematical approximation
- Residual connection around FFN

### CLIP Main Class

```python
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
```
**CLIP architecture:**
- `CLIPEmbedding`: Token + position embeddings
- `12 layers`: Standard CLIP-Base configuration
- `12 heads`: 768 / 12 = 64 dimensions per head
- Final `LayerNorm`: Post-processing normalization

```python
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens) # B, L, D

        # apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            state = layer(state) # (B, L, D)
        output = self.layernorm(state) # (B, L, D)
        
        return output
```
**Forward pass:**
- Ensure token type is correct (LongTensor for embedding lookup)
- Apply embeddings
- Sequential layer processing
- Final normalization
- **Output:** (B, 77, 768) text embeddings for cross-attention

---

## Diffusion U-Net (`diffusion.py`)

### Imports

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention 
```
**Dependencies:**
- Standard PyTorch modules
- Custom attention mechanisms for conditioning

### TimeEmbedding Class

```python
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd * 4)
        self.linear2 = nn.Linear(n_embd * 4, n_embd * 4) # thats why the dimension is 1280 
```
**Time embedding architecture:**
- `n_embd=320`: Input time embedding dimension
- `n_embd * 4 = 1280`: Output dimension
- Two-layer MLP for time conditioning

```python
    def forward(self, x):
        x = self.linear1(x)  # B, 320 -> B, 1280
        x = F.silu(x)  # B, 1280
        x = self.linear2(x)  # B, 1280 -> B, 1280
        return x  # B, 1280 
```
**Time embedding forward:**
- First linear layer: Expand dimension
- SiLU activation: Smooth, non-saturating
- Second linear layer: Refine representation
- **Purpose:** Convert scalar timestep to rich embedding

### Unet_AttentionBlock Class

```python
class Unet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
```
**Attention block setup:**
- `channels`: Total embedding dimension (n_head × n_embd)
- `d_context=768`: CLIP text embedding dimension
- Combines self-attention and cross-attention

```python
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
```
**Input processing:**
- `GroupNorm`: Normalizes channel groups for stable training
- `eps=1e-6`: Small value for numerical stability
- `conv_input`: 1×1 conv for input projection

```python
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
```
**Attention layers:**
- `attention_1`: Self-attention for spatial relationships
- `attention_2`: Cross-attention with text embeddings
- `in_proj_bias=False`: Following standard practice for attention

```python
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
```
**GEGLU feed-forward:**
- `linear_geglu_1`: Expands to 8× channels (4× for value, 4× for gate)
- `linear_geglu_2`: Projects back to original dimension
- **GEGLU:** Gated Linear Unit with GELU activation

```python
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
```
**Output projection:**
- 1×1 conv for final output processing

```python
    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
```
**Input processing:**
- Store long residual connection
- Normalize and project input

```python
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)
```
**Reshape for attention:**
- Flatten spatial dimensions: (B, C, H, W) → (B, C, H×W)
- Transpose: (B, C, H×W) → (B, H×W, C)
- **Purpose:** Treat spatial positions as sequence elements

```python
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
```
**Self-attention block:**
- Short residual around self-attention
- LayerNorm before attention (pre-norm)
- Self-attention across spatial positions

```python
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
```
**Cross-attention block:**
- Another short residual
- Cross-attention with text context
- Enables text conditioning of image generation

```python
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
```
**GEGLU feed-forward:**
- Third short residual
- Split linear output into value and gate
- Apply gated activation: `value * GELU(gate)`
- Project back to original dimension

```python
        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.conv_output(x) + residue_long
```
**Output processing:**
- Reshape back to spatial format
- Apply output convolution
- Add long residual connection

### Unet_ResidualBlock Class

```python
class Unet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
```
**Residual block components:**
- `groupnorm_feature`: Normalizes input features
- `conv_feature`: Main convolution
- `linear_time`: Projects time embedding to feature space
- `n_time=1280`: Time embedding dimension

```python
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
```
**Second convolution and skip connection:**
- Second GroupNorm + Conv for refinement
- **Skip connection logic:**
  - Same channels: Identity (direct connection)
  - Different channels: 1×1 conv projection

```python
    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
```
**Feature processing:**
- Store input for residual
- GroupNorm → SiLU → Conv
- Standard residual block pattern

```python
        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
```
**Time injection:**
- Apply SiLU to time embedding
- Project time to feature space
- `unsqueeze(-1).unsqueeze(-1)`: (B, C) → (B, C, 1, 1)
- Broadcast addition: time added to all spatial positions

```python
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
```
**Final processing:**
- Second GroupNorm → SiLU → Conv on merged features
- Add residual connection (with projection if needed)

### SwitchSequential Class

```python
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, Unet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, Unet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
```
**Smart sequential container:**
- **Purpose:** Handle different layer types with different inputs
- `Unet_AttentionBlock`: Needs context (text embeddings)
- `Unet_ResidualBlock`: Needs time embedding
- **Others:** Standard single-input layers (Conv2d, etc.)

### Upsample Class

```python
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest') # B, C, H, W -> B, C, 2H, 2W
        return self.conv(x)
```
**Upsampling implementation:**
- `F.interpolate(..., mode='nearest')`: Nearest neighbor upsampling
- Followed by 3×3 conv for smoothing
- **Alternative to:** Transposed convolution (less artifacts)

### UNET Main Class

```python
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
```
**Encoder start:**
- Input: Latent representation (4 channels)
- Initial conv: 4 → 320 channels
- Maintains H/8×W/8 spatial resolution

```python
            SwitchSequential(Unet_ResidualBlock(320, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(320, 320), Unet_AttentionBlock(8, 40)),
```
**First encoder stage:**
- 2 blocks with ResidualBlock + AttentionBlock
- `8 heads, 40 dim/head = 320 total`
- Self + cross attention at H/8 resolution

```python
            # (B, 320, H/8, W/8) -> (B, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
```
**First downsampling:**
- Stride-2 conv: Spatial resolution halved
- Maintains 320 channels

```python
            # (B, 320, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(Unet_ResidualBlock(320, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(Unet_ResidualBlock(640, 640), Unet_AttentionBlock(8, 80)),
```
**Second encoder stage:**
- Channel expansion: 320 → 640
- `8 heads, 80 dim/head = 640 total`
- H/16 resolution processing

```python
            # (B, 640, H/16, W/16) -> (B, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
```
**Second downsampling:**
- Another 2× spatial reduction
- Maintains 640 channels

```python
            # (B, 640, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(Unet_ResidualBlock(640, 1280), Unet_AttentionBlock(8, 160)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280), Unet_AttentionBlock(8, 160)),
```
**Third encoder stage:**
- Channel expansion: 640 → 1280
- `8 heads, 160 dim/head = 1280 total`
- H/32 resolution processing

```python
            # (B, 1280, H/32, W/32) -> (B, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280)),
        ])
```
**Final encoder stage:**
- Third downsampling to H/64 resolution
- 2 residual blocks (no attention at lowest resolution)
- Maintains 1280 channels

```python
        self.bottleneck = SwitchSequential(
            Unet_ResidualBlock(1280, 1280),
            Unet_AttentionBlock(8, 160),
            Unet_ResidualBlock(1280, 1280),
        )
```
**Bottleneck processing:**
- Processes at lowest resolution (H/64)
- ResidualBlock → AttentionBlock → ResidualBlock
- Critical for global context understanding

```python
        self.decoders = nn.ModuleList([
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(Unet_ResidualBlock(2560, 1280)),
```
**Decoder start:**
- Input: 2560 channels (1280 from bottleneck + 1280 from skip)
- **Skip connection:** Concatenates encoder features
- Output: 1280 channels

```python
            SwitchSequential(Unet_ResidualBlock(2560, 1280)),
            SwitchSequential(Unet_ResidualBlock(2560, 1280), Upsample(1280)),
```
**First decoder blocks:**
- More 2560 → 1280 processing
- Final block includes upsampling: H/64 → H/32

```python
            # (B, 2560, H/32, W/32) -> (B, 1280, H/32, W/32)
            SwitchSequential(Unet_ResidualBlock(2560, 1280), Unet_AttentionBlock(8, 160)),
            SwitchSequential(Unet_ResidualBlock(2560, 1280), Unet_AttentionBlock(8, 160)),
            # (B, 1920, H/32, W/32) -> (B, 1280, H/16, W/16)
            SwitchSequential(Unet_ResidualBlock(1920, 1280), Unet_AttentionBlock(8, 160), Upsample(1280)),
```
**Second decoder stage:**
- Process at H/32 with attention
- **Note:** Input changes to 1920 (1280 + 640 from skip)
- Final block upsamples: H/32 → H/16

```python
            # (B, 1920, H/16, W/16) -> (B, 640, H/16, W/16)
            SwitchSequential(Unet_ResidualBlock(1920, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(Unet_ResidualBlock(1280, 640), Unet_AttentionBlock(8, 80)),
            # (B, 960, H/16, W/16) -> (B, 640, H/8, W/8)
            SwitchSequential(Unet_ResidualBlock(960, 640), Unet_AttentionBlock(8, 80), Upsample(640)),
```
**Third decoder stage:**
- Channel reduction: → 640
- **Skip connection math:** 1920 = 1280 + 640, then 1280 = 640 + 640
- Then 960 = 640 + 320 from encoder skip
- Upsample: H/16 → H/8

```python
            # (B, 960, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(Unet_ResidualBlock(960, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(640, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(640, 320), Unet_AttentionBlock(8, 40)),
        ])
```
**Final decoder stage:**
- Channel reduction: → 320
- **Skip connections:** 960 = 640 + 320, then 640 = 320 + 320
- Final output: 320 channels at H/8 resolution

### UNET Forward Pass

```python
    def forward(self, x, context, time):
        # x: (B, 4, H/8, W/8)
        # context: (B, L, D)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
```
**Encoder forward pass:**
- Process through each encoder block
- Store outputs for skip connections
- Each layer gets image features, text context, and time

```python
        # x: (B, 1280, H/64, W/64)
        x = self.bottleneck(x, context, time)
        # x: (B, 1280, H/64, W/64)
```
**Bottleneck processing:**
- Process at lowest resolution
- Critical for global understanding

```python
        for layers in self.decoders:
            # concatenate with the corresponding skip connection
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        
        # x: (B, 320, H/8, W/8)
        return x
```
**Decoder forward pass:**
- `skip_connections.pop()`: Get last encoder output (LIFO order)
- `torch.cat(..., dim=1)`: Concatenate along channel dimension
- Process through decoder blocks
- Final output: (B, 320, H/8, W/8)

### UNET_OutputLayer Class

```python
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x
```
**Output processing:**
- GroupNorm + SiLU + Conv
- Projects from U-Net features to final output
- Typically: 320 → 4 channels (same as input latent)

### Diffusion Main Class

```python
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
```
**Complete diffusion model:**
- `TimeEmbedding`: 320 → 1280 time conditioning
- `UNET`: Main denoising network
- `UNET_OutputLayer`: 320 → 4 final projection

```python
    def forward(self, latent, context, time):
        # latent: B, 4, H/8, W/8
        # context: B, L, D
        # time: 1, 320
        
        time_emb = self.time_embedding(time) # 1, 320 -> 1, 1280
        
        output = self.unet(latent, context, time_emb)  # B, 4, H/8, W/8 -> B, 320, H/8, W/8 
        
        output = self.final(output) # B, 320, H/8, W/8 -> B, 4, H/8, W/8 
        
        return output # B, 4, H/8, W/8
```
**Complete forward pass:**
1. **Time embedding:** Convert timestep to rich representation
2. **U-Net processing:** Main denoising with text conditioning
3. **Final projection:** Convert back to latent space
4. **Output:** Predicted noise or denoised latent

---

## Summary of Key Implementation Details

### Critical Design Patterns

1. **Skip Connections:** U-Net concatenates encoder features to decoder
2. **Multi-Head Attention:** Parallel processing with different learned projections
3. **Cross-Attention:** Enables text conditioning in image generation
4. **Residual Connections:** Short and long residuals for gradient flow
5. **Group Normalization:** Stable training across varying batch sizes
6. **Time Injection:** Broadcast time embeddings to all spatial positions

### Mathematical Foundations

1. **Reparameterization Trick:** `z = μ + σ * ε` for differentiable sampling
2. **Scaled Dot-Product Attention:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
3. **GEGLU Activation:** `x * GELU(gate)` for improved gradients
4. **Asymmetric Padding:** Ensures proper dimensions with stride-2 convolutions

### Implementation Issues Found

1. **Attention Class:** `self.d_ehead` should be `self.d_head`
2. **VAE Residual Blocks:** `padding=0` causes spatial mismatch with residual
3. **CLIP Layer:** `casual_mask` should be `causal_mask`
4. **VAE Attention:** Should specify n_heads in SelfAttention constructor

### Architectural Significance

This implementation demonstrates a complete understanding of:
- **Latent Diffusion:** Operating in compressed latent space
- **Conditional Generation:** Text-guided image synthesis
- **Multi-Modal Learning:** Combining vision and language models
- **Progressive Processing:** Multi-scale feature learning in U-Net
- **Attention Mechanisms:** Both self and cross-attention for different purposes

The code represents a sophisticated deep learning system that combines multiple state-of-the-art techniques into a cohesive generative model.
