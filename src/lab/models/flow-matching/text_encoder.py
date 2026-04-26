import torch

import torch.nn as nn



class TextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=1024, max_len=96):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # 用一个简单的 Transformer 层模拟复杂的文本语义提取
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=2
        )

    def forward(self, input_ids):

        x = self.token_embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1), :]
        x = self.transformer(x)
        return x # 输出 [B, L, 1024] 供 DiT 做 Cross-Attention