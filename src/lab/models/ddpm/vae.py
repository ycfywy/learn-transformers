import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    轻量 VAE（与 flow-matching 版本完全相同，方便对比实验）

    Encoder: [B, 3, 256, 256] → [B, 4, 32, 32]   (下采样 8×)
    Decoder: [B, 4, 32, 32]  → [B, 3, 256, 256]  (上采样 8×)
    """

    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),   # 256→128
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # 128→64
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, stride=2, padding=1),  # 64→32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),              # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),      # 128→256
            nn.Tanh()   # 输出范围 [-1, 1]，与数据归一化对齐
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
