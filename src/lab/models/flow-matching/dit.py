
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

from .layers import PatchEmbed, TimeStepEmbedder


class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        
        
        self.mlp = nn.Sequential(
            
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        ) 

    def forward(self, x, c: torch.Tensor, context: torch.Tensor):
        """
        x: [B, N, D]
        c: time embedding [B, T]
        context: [B, L, D]


        output: [B, N, D]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            cast(torch.Tensor, self.adaLN_modulation(c)).chunk(6, dim=1)
        
        res = x
        x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # TODO 这里不太懂 nn.MuitiHeadAttention怎么用
        x, _ = self.attn(x, x, x)
        x = res + gate_msa.unsqueeze(1) * x

        # cross attention (x as q, context as k v)
        x = x + self.cross_attn(self.norm_cross(x), context, context)[0]


        res = x
        x = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = self.mlp(x)

        x = res + gate_mlp.unsqueeze(1) * x

        return x


class DiT(nn.Module):
    def __init__(self, 
                 input_size=32, 
                 patch_size=2, 
                 in_channels=4, 
                 hidden_size=768,
                 depth=12,
                 num_heads=12,
                 context_dim=1024):
        super().__init__()

        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimeStepEmbedder(hidden_size)

        self.y_proj = nn.Linear(context_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, \
                        self.patch_embed.num_patches, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        )
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_size = input_size
    
    def forward(self, x, t, context):
        """
        x: [B, C, H, W], t: [B], context: [B, L, context_dim]

        before unpatchfy:[B, N,  patch_size * patch_size * in_channels]
                 N = H  * W / (patch_size * patch_size) 
        after unpatchfy: [B, C, H, W]      
        """   
        x = self.patch_embed(x) + self.pos_embed
        t_embed = self.t_embedder(t)
        context = self.y_proj(context)

        for block in self.blocks:
            x = block(x, t_embed, context)

        x = self.final_layer(x)
        return self.unpatchfy(x)

    def unpatchfy(self, x: torch.Tensor):
        
        p  = self.patch_size
        h = w = int((x.shape[1] ** 0.5))

        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)

        imgs = x.reshape(x.shape[0], self.in_channels, h * p, w * p)

        return imgs


        

