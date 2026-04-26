import torch
import torch.nn as nn
import math



class TimeStepEmbedder(nn.Module):
    """_summary_

        将输入的时间步t转换为固定长度 frequency_embedding_size 的 vector
        并经过一个网络，映射转换到hidden_size长度

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
        输出一个长度为 dim 的正余弦编码
        
        不过这里是把 cos放在前 dim / 2的位置上 sin放在后面
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half) 
        
        # apply sin in 0 2 4 6 ... pos
        # apply cos in 1 3 5 6 ... pos
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
    input:    [B, 4, 32, 32] (Batch, Channels, Height, Width)
    output:   [B, 16 * 16, 768] (假设 patch_size=2)

    """

    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor):
        print("before patch", x.shape)
        x = self.proj(x)
        #  from dim 2, flatten
        # [B, D, h, w] ->  [B, N, H]
        x = x.flatten(2).transpose(1, 2)
        print("after patch", x.shape)
        return x

