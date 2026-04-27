import os

import torch
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from .pokemon_dataset import PokemonDataset
from .vae import VAE
from .text_encoder import TextEncoder
from .dit import DiT
from .ddpm_dit import DDPMDiT


# ==========================================
# 1. 配置管理中心
# ==========================================
class Config:
    # 路径配置
    split = "train[:10]"
    save_dir = "./outputs_ddpm"
    cache_dir = "./data"

    # 训练超参数
    batch_size = 4
    lr = 1e-4
    epochs = 10
    save_interval = 1

    # DDPM 噪声调度
    T = 1000                # 扩散总步数
    beta_start = 1e-4
    beta_end = 0.02

    # 采样步数（DDPM 原版 = T；加速采样可用 DDIM，这里先用 T 步）
    sample_steps = 1000

    # 模型架构
    vocab_size = 1000
    latent_size = 32
    latent_channels = 4
    text_embed_dim = 1024
    hidden_size = 768

    # 硬件
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. 训练主程序
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # --- 数据集 ---
    my_vocab = {"<pad>": 0, "unknown": 1}
    dataset = PokemonDataset(
        split=cfg.split,
        vocab=my_vocab,
        cache_dir=cfg.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=1,
    )

    # 固定用于可视化的样本
    fixed_indices = [0, 1, 2, 3]
    fixed_caps = torch.stack([dataset[i][1] for i in fixed_indices]).to(cfg.device)

    # --- 初始化组件 ---
    my_vae = VAE(in_channels=3, latent_channels=cfg.latent_channels).to(cfg.device)
    my_text_enc = TextEncoder(
        vocab_size=cfg.vocab_size, embed_dim=cfg.text_embed_dim
    ).to(cfg.device)
    my_dit = DiT(
        input_size=cfg.latent_size,
        patch_size=2,
        in_channels=cfg.latent_channels,
        hidden_size=cfg.hidden_size,
        depth=1,          # 快速实验：只用 1 层；正式训练建议 12 层
    ).to(cfg.device)

    model = DDPMDiT(
        dit_model=my_dit,
        vae_model=my_vae,
        text_encoder=my_text_enc,
        T=cfg.T,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(cfg.device)

    # --- 优化器（只更新 DiT 参数）---
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=cfg.lr)

    print(f"🚀 开始训练 DDPM | 设备: {cfg.device} | Batch Size: {cfg.batch_size} | T: {cfg.T}")

    # --- 训练循环 ---
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for imgs, caps in progress_bar:
            imgs, caps = imgs.to(cfg.device), caps.to(cfg.device)

            optimizer.zero_grad()

            # Forward: VAE 编码 → 加噪 → DiT 预测 ε → MSE Loss
            loss = model(imgs, caps)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} 平均 Loss: {avg_loss:.4f}")

        # --- 定期采样与保存 ---
        if epoch % cfg.save_interval == 0:
            model.eval()
            print(f"\n📸 正在生成 Epoch {epoch} 的预览图（需跑完 {cfg.sample_steps} 去噪步）...")

            with torch.no_grad():
                # DDPM 完整去噪采样（速度较慢，可减小 sample_steps 快速预览）
                generated_imgs = model.sample(fixed_caps, steps=cfg.sample_steps)

                # 逆标准化 [-1,1] → [0,1]
                preview = (generated_imgs * 0.5 + 0.5).clamp(0, 1)

                img_path = os.path.join(cfg.save_dir, f"sample_epoch_{epoch}.png")
                utils.save_image(preview, img_path, nrow=2)

                txt_path = os.path.join(cfg.save_dir, f"sample_epoch_{epoch}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    for i in range(len(fixed_indices)):
                        ids = fixed_caps[i].tolist()
                        text = "".join([str(my_vocab.get(idx, "")) for idx in ids if idx > 1])
                        f.write(f"Image {i}: {text}\n")

                ckpt_path = os.path.join(cfg.save_dir, f"dit_epoch_{epoch}.pth")
                torch.save(model.dit.state_dict(), ckpt_path)

            print(f"✅ Epoch {epoch} 记录完成：")
            print(f"   🖼️  图像: {img_path}")
            print(f"   📝  文本: {txt_path}")
            print(f"   💾  权重: {ckpt_path}")


if __name__ == "__main__":
    main()
