# src/models/embeddings.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        dim: 嵌入的维度，必须是偶数（因为要平分给 sin 和 cos）
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        输入 time: 形状为 [batch_size] 的张量，包含当前的时间步 t
        输出: 形状为 [batch_size, dim] 的特征向量
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 计算缩放因子：10000^(2i/dim)
        # 这里的数学逻辑是为了让不同维度的频率不同，从而捕捉长短期的依赖
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 矩阵相乘：time[B, 1] * embeddings[1, D/2] -> [B, D/2]
        embeddings = time[:, None] * embeddings[None, :]
        
        # 拼接 sin 和 cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings