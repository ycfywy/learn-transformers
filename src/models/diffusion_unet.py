# src/models/diffusion_unet.py
import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings # 之前写的文件

class SimpleUnet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        # 时间编码 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.ReLU()
        )
        
        # 极简版 U-Net 结构
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Down (下采样)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 3, padding=1, stride=2)
        
        # Middle
        self.mid_block = nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)
        
        # Up (上采样)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, 2, 1)
        
        # Final
        self.final_conv = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        
        # 简单融合时间信息 (这里简化处理，实际应用中会加在 ResBlock 里)
        x2 = x2 + t_emb[:, :, None, None] 
        
        x3 = self.mid_block(x2)
        x4 = self.up1(x3)
        
        return self.final_conv(x4 + x1) # 残差连接