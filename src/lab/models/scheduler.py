import torch


"""
DDPM 加噪声的过程
"""


class DDPMScheduler:

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x_start: torch.Tensor, t):
        """
        x_start: 原始样本, shape = [B, C, H, W] 
        t: 每个样本对应的时间步, shape = [B]
        """   
        noise = torch.randn_like(x_start)
        alpha_cumprod_t = self.alphas_cumprod.to(x_start.device)[t]
        alpha_cumprod_t_boardcast = alpha_cumprod_t.view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_cumprod_t_boardcast) * x_start + torch.sqrt(1.0 - alpha_cumprod_t_boardcast) * noise

        return x_t, noise
    

if __name__ == "__main__":
    scheduler = DDPMScheduler()
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,), dtype=torch.long)

    x_t, noise = scheduler.add_noise(x_start, t)

    print("x_start.shape =", x_start.shape)
    print("x_t.shape =", x_t.shape)
    print("noise.shape =", noise.shape)
    print("t =", t)