import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)  # B, L, 3 * d_embed
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_ehead = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: B, L, d_embed
        batch_size, seq_len, d_embed = x.shape
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # Project input to query, key, value
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # B, L, D -> B, L, dim*3 -> 3*(B, L, dim)
        
        q = q.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head
        k = k.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head
        v = v.view(interm_shape).transpose(1, 2)  # B, L, D -> B, L, H, d_head -> B, H, L, d_head

        weight = q @ k.transpose(-1, -2)  # B, H, L, d_head @ B, H, d_head, L -> B, H, L, L
        
        if causal_mask: 
            #upper traingle is made up of 1 then fill up with -infinity(only upper traingle though)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            weight.masked_fill_(mask, float('-inf'))  # Set upper triangle to -inf  
            
        weight = weight / (self.d_head ** 0.5)  # Scale by sqrt(d_head) 
        
        weight = weight.softmax(dim=-1)  # B, H, L, L -> B, H, L, L
        
        output = weight @ v  # B, H, L, L @ B, H, L, d_head -> B, H, L, d_head
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_embed)  # B, H, L, d_head -> B, L, H, d_head -> B, L, D
        output = self.out_proj(output)  # B, L, D
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (Query): B, L_q, D_embed
        # y (Key, Value): B, L_kv, D_cross
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        # B, L_kv, H, d_head
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project query, key, value
        q = self.q_proj(x) # B, L_q, D_embed
        k = self.k_proj(y) # B, L_kv, D_embed
        v = self.v_proj(y) # B, L_kv, D_embed

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2) # B, H, L_q, d_head
        k = k.view(interim_shape).transpose(1, 2) # B, H, L_kv, d_head
        v = v.view(interim_shape).transpose(1, 2) # B, H, L_kv, d_head

        # Calculate attention scores
        weight = q @ k.transpose(-1, -2) # B, H, L_q, d_head @ B, H, d_head, L_kv -> B, H, L_q, L_kv
        
        # Scale
        weight /= math.sqrt(self.d_head)
        
        # Softmax
        weight = F.softmax(weight, dim=-1) # B, H, L_q, L_kv

        # Apply attention to value
        output = weight @ v # B, H, L_q, L_kv @ B, H, L_kv, d_head -> B, H, L_q, d_head
        
        # Reshape and transpose back
        output = output.transpose(1, 2).contiguous() # B, L_q, H, d_head
        output = output.view(input_shape) # B, L_q, D_embed
        
        # Final projection
        output = self.out_proj(output) # B, L_q, D_embed
        return output