import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        """
            h_out = ( h_in + 2 * padding - kernel ) / stride + 1
        """

        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        
            nn.Conv2d(128, latent_channels, 3, stride=2, padding=1)
        )

        # [B, 4, H/8, W/8]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, \
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            # TODO 为什么还要激活
            # 1. 像素归一化 dl中其实输入的是[-1, 1]为了保证输出也是 可以用tanh 
            # 2. tanh在 -1 1 接近的时候会比较平缓 不至于像素极端纯黑纯白
            nn.Tanh()          
        )   
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output