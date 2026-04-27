import torch
import torch.nn as nn
from typing import cast

from .layers import PatchEmbed, TimeStepEmbedder


class DiTBlock(nn.Module):
    """
    单个 DiT Block，结构与 flow-matching 版本完全相同：
      Self-Attention  (adaLN-Zero 调制)
      Cross-Attention (文本 context)
      FFN             (adaLN-Zero 调制)

    DDPM 与 Flow-Matching 的区别只在"预测目标"上（见 DDPMDiT），
    DiT Block 本身不需要改动。
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-Attention
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        # adaLN-Zero：从时间嵌入预测 6 个调制参数（shift/scale/gate × 2）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor):
        """
        x:       [B, N, D]   patch 序列
        c:       [B, D]      时间步嵌入
        context: [B, L, D]   文本特征（已投影到 hidden_size）
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            cast(torch.Tensor, self.adaLN_modulation(c)).chunk(6, dim=1)

        # --- Self-Attention ---
        res = x
        x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x, _ = self.attn(x, x, x)
        x = res + gate_msa.unsqueeze(1) * x

        # --- Cross-Attention (x 作 Query, context 作 K/V) ---
        x = x + self.cross_attn(self.norm_cross(x), context, context)[0]

        # --- FFN ---
        res = x
        x = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = self.mlp(x)
        x = res + gate_mlp.unsqueeze(1) * x

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) backbone。

    与 flow-matching 版本架构完全一致，只是在 DDPM 中
    forward 的输出含义变为"预测噪声 epsilon"（而非速度场 v）。
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        context_dim: int = 1024,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimeStepEmbedder(hidden_size)
        self.y_proj = nn.Linear(context_dim, hidden_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        # 最终输出 patch 维度：patch_size² × in_channels
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels),
        )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.input_size = input_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        x:       [B, C, H, W]  noisy latent
        t:       [B]           整数时间步 (0 ~ T-1)
        context: [B, L, context_dim]  文本特征

        返回:    [B, C, H, W]  预测噪声 epsilon_theta
        """
        x = self.patch_embed(x) + self.pos_embed        # [B, N, D]
        t_emb = self.t_embedder(t)                       # [B, D]
        ctx = self.y_proj(context)                       # [B, L, D]

        for block in self.blocks:
            x = block(x, t_emb, ctx)

        x = self.final_layer(x)                          # [B, N, p²C]
        return self._unpatchify(x)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, N, p²C] → [B, C, H, W]
        N = (H/p) * (W/p)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_channels, h * p, w * p)
