import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings


# ──────────────────────────────────────────────
# 基础模块
# ──────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    残差块，含时间步嵌入注入。
    结构：GroupNorm → SiLU → Conv → (time注入) → GroupNorm → SiLU → Dropout → Conv + residual
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x) + self.time_proj(t_emb)[:, :, None, None]
        return self.block2(h) + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """空间自注意力块：GroupNorm → flatten → MHA → unflatten + residual"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)     # [B, HW, C]
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class Downsample(nn.Module):
    """步长为 2 的卷积，分辨率减半，通道数不变。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """nearest 上采样 ×2，再卷积对齐通道。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 组合模块
# ──────────────────────────────────────────────

class DownBlock(nn.Module):
    """编码器单元：2×ResBlock + Downsample，每个 ResBlock 输出压入 skips 栈。"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.res1 = ResBlock(in_ch,  out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.down = Downsample(out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skips: list) -> torch.Tensor:
        x = self.res1(x, t_emb); skips.append(x)
        x = self.res2(x, t_emb); skips.append(x)
        return self.down(x)


class BottleneckBlock(nn.Module):
    """瓶颈层：ResBlock → AttentionBlock → ResBlock。"""
    def __init__(self, channels: int, time_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim, dropout)
        self.attn = AttentionBlock(channels, num_heads)
        self.res2 = ResBlock(channels, channels, time_dim, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.res2(self.attn(self.res1(x, t_emb)), t_emb)


class UpBlock(nn.Module):
    """解码器单元：Upsample + 2×(cat skip → ResBlock)，从 skips 栈 pop 两次。"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.up   = Upsample(in_ch, in_ch)
        self.res1 = ResBlock(in_ch * 2, in_ch,  time_dim, dropout)
        self.res2 = ResBlock(in_ch * 2, out_ch, time_dim, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skips: list) -> torch.Tensor:
        x = torch.cat([self.up(x),         skips.pop()], dim=1)
        x = torch.cat([self.res1(x, t_emb), skips.pop()], dim=1)
        return self.res2(x, t_emb)


# ──────────────────────────────────────────────
# 标准 UNet（DDPM 风格）
# ──────────────────────────────────────────────

class UNet(nn.Module):
    """
    标准扩散模型 UNet。

    结构概览（默认 channel_mults=(1, 2, 4)）：
        init_conv → DownBlock×N → BottleneckBlock → UpBlock×N → final_conv

    Skip 连接：DownBlock 每层输出 2 个 skip，UpBlock 逆序 pop 消费。
    时间嵌入：Sinusoidal → Linear → GELU → Linear（维度 base_dim×4）。

    Args:
        in_channels:   图像通道数，默认 3
        base_dim:      基础通道数，默认 64
        channel_mults: 各编解码层通道倍率，默认 (1, 2, 4)
        num_heads:     Attention 头数，默认 4
        dropout:       ResBlock dropout，默认 0.1
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        time_dim     = base_dim * 4
        channels     = [base_dim * m for m in channel_mults]   # e.g. [64, 128, 256]
        rev_channels = list(reversed(channels))                 # e.g. [256, 128, 64]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Encoder：逐级扩通道，分辨率减半
        enc_in  = [base_dim] + channels[:-1]        # [64,  64, 128]
        self.down_blocks = nn.ModuleList([
            DownBlock(in_ch, out_ch, time_dim, dropout)
            for in_ch, out_ch in zip(enc_in, channels)
        ])

        self.bottleneck = BottleneckBlock(channels[-1], time_dim, num_heads, dropout)

        # Decoder：逐级缩通道，分辨率翻倍
        dec_out = rev_channels[1:] + [base_dim]     # [128,  64, 64]
        self.up_blocks = nn.ModuleList([
            UpBlock(in_ch, out_ch, time_dim, dropout)
            for in_ch, out_ch in zip(rev_channels, dec_out)
        ])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 带噪图像
            t: [B]           整数时间步
        Returns:
            [B, C, H, W] 预测噪声
        """
        t_emb = self.time_mlp(t)
        x0    = self.init_conv(x)

        skips, h = [], x0
        for block in self.down_blocks:
            h = block(h, t_emb, skips)

        h = self.bottleneck(h, t_emb)

        for block in self.up_blocks:
            h = block(h, t_emb, skips)

        return self.final_conv(h + x0)
