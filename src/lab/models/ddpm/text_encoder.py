import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    简单文本编码器：token embedding + positional embedding + Transformer
    输出 [B, L, embed_dim]，供 DiT 做 Cross-Attention。

    与 flow-matching 版本完全相同。
    """

    def __init__(self, vocab_size=1000, embed_dim=1024, max_len=96):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=2,
        )

    def forward(self, input_ids):
        """
        input_ids: [B, L]
        返回:      [B, L, embed_dim]
        """
        x = self.token_embedding(input_ids) + self.pos_embedding[:, : input_ids.size(1), :]
        return self.transformer(x)
