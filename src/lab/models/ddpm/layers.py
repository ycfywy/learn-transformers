import torch
import torch.nn as nn
import math


class TimeStepEmbedder(nn.Module):
    """
    将整数时间步 t (0 ~ T-1) 转换为 sinusoidal 频率特征，
    再经过一个两层 MLP 映射到 hidden_size 维的向量。

    flow-matching 版本里 t 是 [0, 1] 的连续值；
    DDPM 版本里 t 是离散整数 [0, T-1]，但 embedding 方式完全一样。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Sinusoidal positional encoding（DDPM / Transformer 常用）
        t: [B]  整数或浮点时间步
        返回: [B, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PatchEmbed(nn.Module):
    """
    把 latent 图像 [B, C, H, W] 切成 patch 序列 [B, N, D]

    例: input_size=32, patch_size=2, in_chans=4, embed_dim=768
        → num_patches = (32/2)^2 = 256
        → output [B, 256, 768]
    """

    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor):
        x = self.proj(x)           # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
