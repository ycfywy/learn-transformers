"""
DDPMDiT —— DDPM 扩散模型 + DiT backbone

与 FlowDiT (flow-matching) 的核心区别：
┌───────────────────┬──────────────────────────────┬──────────────────────────────────┐
│                   │  Flow Matching               │  DDPM                            │
├───────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 时间步            │ t ~ U(0, 1) 连续值           │ t ~ U{0, T-1} 离散整数           │
│ 加噪过程          │ x_t = (1-t)x_0 + t·ε         │ x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε    │
│ 预测目标          │ 速度场 v = x_1 - x_0          │ 噪声 ε (epsilon prediction)      │
│ 推理方式          │ Euler ODE 积分 (1 路径)       │ DDPM 逐步去噪 (T 步 Markov 链)    │
└───────────────────┴──────────────────────────────┴──────────────────────────────────┘

加噪公式（前向过程）：
    q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 - ᾱ_t) · I)
    ᾱ_t = ∏_{s=1}^{t} α_s,  α_s = 1 - β_s

去噪损失（训练）：
    L = E_{t,x_0,ε} [ ||ε - ε_θ(x_t, t, context)||² ]

逆向采样（推理，DDPM 原始公式）：
    x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
    σ_t = √β_t,  z ~ N(0, I)  (t > 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE
from .dit import DiT
from .text_encoder import TextEncoder


def make_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    """线性 beta schedule，DDPM 原始论文设定。"""
    return torch.linspace(beta_start, beta_end, T)


class DDPMDiT(nn.Module):
    """
    DDPM + DiT 文生图模型。

    组件与 FlowDiT 一一对应：
      - vae         : 图像 ↔ latent 编解码（训练时冻结）
      - text_encoder: 文本 → context 向量（训练时冻结）
      - dit         : 主干网络，预测噪声 ε_θ(x_t, t, context)
    """

    # register_buffer 注册的张量，显式声明类型供静态分析器识别
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    betas_over_sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(
        self,
        dit_model: DiT,
        vae_model: VAE,
        text_encoder: TextEncoder,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.dit = dit_model
        self.vae = vae_model
        self.text_encoder = text_encoder
        self.T = T

        # 冻结 VAE 与 TextEncoder（只训练 DiT）
        self.vae.eval()
        self.text_encoder.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # ---- DDPM 噪声调度表（register_buffer 随模型一起转移设备）----
        betas = make_beta_schedule(T, beta_start, beta_end)          # [T]
        alphas = 1.0 - betas                                          # [T]
        alphas_cumprod = torch.cumprod(alphas, dim=0)                 # ᾱ_t, [T]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 加噪所需系数
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # 逆向采样所需系数
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        # β_t / √(1 - ᾱ_t)
        self.register_buffer(
            "betas_over_sqrt_one_minus_alphas_cumprod",
            betas / (1.0 - alphas_cumprod).sqrt(),
        )
        # 后验方差 σ_t² = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _extract(self, coeff: torch.Tensor, t: torch.Tensor, shape):
        """从 1D 调度表中按时间步索引取值，并广播到 shape。"""
        out = coeff[t]                              # [B]
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        前向加噪（封闭形式）：
          x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # 训练 Forward
    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor):
        """
        训练阶段：
          1. VAE 编码图像 → latent x_0
          2. 采样随机时间步 t 和噪声 ε
          3. 前向加噪得到 x_t
          4. DiT 预测 ε_θ(x_t, t, context)
          5. MSE Loss

        pixel_values: [B, 3, 256, 256]
        input_ids:    [B, L]
        返回: scalar loss
        """
        device = pixel_values.device
        B = pixel_values.shape[0]

        # Step 1: 编码到 latent 空间
        with torch.no_grad():
            x_0 = self.vae.encoder(pixel_values)          # [B, 4, 32, 32]
            context = self.text_encoder(input_ids)         # [B, L, 1024]

        # Step 2: 随机采样时间步和噪声
        t = torch.randint(0, self.T, (B,), device=device).long()   # [B]
        noise = torch.randn_like(x_0)                               # ε ~ N(0, I)

        # Step 3: 加噪
        x_t = self.q_sample(x_0, t, noise)                # [B, 4, 32, 32]

        # Step 4: DiT 预测噪声
        noise_pred = self.dit(x_t, t.float(), context)    # [B, 4, 32, 32]

        # Step 5: MSE 损失（预测目标是真实噪声 ε）
        loss = F.mse_loss(noise_pred, noise)
        return loss

    # ------------------------------------------------------------------
    # 推理采样（DDPM 逆向过程）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor, steps: int | None = None):
        """
        从纯噪声出发，逐步去噪 T 步，最后 VAE 解码为图像。

        input_ids: [B, L]
        steps:     去噪步数，默认等于训练时的 T（也可设为小值加速，
                   但 DDPM 原版不支持跳步，跳步请用 DDIM）
        返回: [B, 3, 256, 256]，像素值范围 [-1, 1]
        """
        self.dit.eval()
        steps = steps or self.T
        B = input_ids.shape[0]
        device = input_ids.device

        # 文本 context
        context = self.text_encoder(input_ids)   # [B, L, 1024]

        # 从标准正态噪声出发 x_T
        x = torch.randn(
            B, self.dit.in_channels, self.dit.input_size, self.dit.input_size,
            device=device,
        )

        # 逐步去噪：t = T-1 → 0
        for i in reversed(range(steps)):
            t_batch = torch.full((B,), i, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.dit(x, t_batch.float(), context)   # [B, C, H, W]

            # 计算 x_{t-1} 均值
            coeff = self._extract(self.betas_over_sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            sqrt_recip = self._extract(self.sqrt_recip_alphas, t_batch, x.shape)
            mean = sqrt_recip * (x - coeff * noise_pred)

            if i > 0:
                # 采样随机方差（后验）
                var = self._extract(self.posterior_variance, t_batch, x.shape)
                z = torch.randn_like(x)
                x = mean + var.sqrt() * z
            else:
                # 最后一步不加噪声
                x = mean

        # VAE 解码
        samples = self.vae.decoder(x)
        return samples
