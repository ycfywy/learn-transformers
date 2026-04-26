import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .vae import VAE
from .dit import DiT
from .text_encoder import TextEncoder

class FlowDiT(nn.Module):
    def __init__(self, dit_model, vae_model, text_encoder):
        super().__init__()
        self.dit = dit_model
        self.vae = vae_model
        self.text_encoder = text_encoder
        
        # 冻结 VAE 和 TextEncoder 的参数（通常文生图只练 DiT）
        self.vae.eval()
        self.text_encoder.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def get_velocity_target(self, x_1, x_0, t):
        """
        Flow Matching 核心公式：x_t = (1 - t) * x_0 + t * x_1
        目标速度场 (Target Velocity) v = dx_t / dt = x_1 - x_0
        """
        t_pad = t.view(-1, 1, 1, 1)
        x_t = (1 - t_pad) * x_0 + t_pad * x_1
        target = x_1 - x_0
        return x_t, target

    def forward(self, pixel_values, input_ids):
        """
        训练阶段的 Forward
        pixel_values: [B, 3, 256, 256]
        input_ids: [B, L]
        """
        device = pixel_values.device
        B = pixel_values.shape[0]

        # 1. 将图像编码到 Latent 空间 [B, 4, 32, 32]
        print("before vae encoder", pixel_values.shape)
        with torch.no_grad():
            x_1 = self.vae.encoder(pixel_values)
            # 提取文本特征 [B, L, 1024]
            context = self.text_encoder(input_ids)
        print("after vae", x_1.shape)
        # 2. 采样时间步 t ~ U(0, 1) 和噪声 x_0
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)

        # 3. 构造 Flow 路径
        x_t, target = self.get_velocity_target(x_1, x_0, t)

        # 4. DiT 预测速度场 v_theta
        v_pred = self.dit(x_t, t, context)

        # 5. 计算损失 (MSE)
        loss = F.mse_loss(v_pred, target)
        return loss

    @torch.no_grad()
    def sample(self, input_ids, steps=50, cfg_scale=7.5):
        """
        推理阶段：从噪声生成图像
        """
        self.dit.eval()
        B = input_ids.shape[0]
        device = input_ids.device
        
        # 1. 准备 Context 和 初始噪声
        context = self.text_encoder(input_ids)
        # 如果要支持 CFG，这里需要 null_context (省略展示)
        
        x = torch.randn(B, self.dit.in_channels, self.dit.input_size, self.dit.input_size).to(device)
        dt = 1.0 / steps

        # 2. Euler 积分 ODE
        for i in range(steps):
            t_curr = i / steps
            t = torch.ones(B, device=device) * t_curr
            
            # 预测当前斜率 (速度)
            v = self.dit(x, t, context)
            
            # x_{t+dt} = x_t + v * dt
            x = x + v * dt

        # 3. VAE 解码回到像素
        samples = self.vae.decoder(x)
        return samples




