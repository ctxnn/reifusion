import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embd, n_tokens):
        
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        x = self.token_embedding(tokens)  # (B, L, D)
        x += self.position_embedding
        return x 
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)


    def forward(self, x):
        # B, L, D = x.shape

        residue = x
        x = self.layernorm1(x)  # layer normalization
        attn_output = self.attention(x, casual_mask=True)  # (B, L, D)
        x += residue  # residual connection
        
        residue = x
        x = self.layernorm2(x)  # layer normalization

        # Feed-forward network
        x = self.linear1(x)  # (B, L, 4D)
        x = x * torch.sigmoid(1.702 * x)  # quick gelu activation
        x = self.linear2(x)  # (B, L, D)
        x += residue  # residual connection
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens) # B, L, D

        # apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            state = layer(state) # (B, L, D)
        output = self.layernorm(state) # (B, L, D)
        
        return output