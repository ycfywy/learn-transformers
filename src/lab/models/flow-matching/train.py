import os
import sys




import torch
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm


from .pokemon_dataset import PokemonDataset
from .flow_dit import FlowDiT, VAE, TextEncoder, DiT




# ==========================================
# 1. 配置管理中心
# ==========================================
class Config:
    # 路径配置
    split = "train[:10]"           # 数据集切片
    save_dir = "./outputs"         # 模型与样本保存目录
    cache_dir = "./data"           # 数据集缓存目录
    
    # 训练超参数
    batch_size = 4
    lr = 1e-4
    epochs = 10
    sample_steps = 30              # 采样时的 ODE 步数
    save_interval = 1              # 每隔多少 Epoch 保存/采样
    
    # 模型架构参数 
    vocab_size = 1000
    latent_size = 32               
    latent_channels = 4
    text_embed_dim = 1024
    hidden_size = 768
    
    # 硬件设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 训练主程序
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("not start ? ")
    my_vocab = {"<pad>": 0, "unknown": 1} # 示例，请使用你实际定义的词表
    
    dataset = PokemonDataset(
        split=cfg.split, 
        vocab=my_vocab, 
        cache_dir=cfg.cache_dir
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=1
    )

    fixed_indices = [0, 1, 2, 3] 
    fixed_imgs_list = []
    fixed_caps_list = []
    for idx in fixed_indices:
        img_t, cap_t = dataset[idx]
        fixed_imgs_list.append(img_t)
        fixed_caps_list.append(cap_t)
    
    # 堆叠成 batch 并移到设备
    fixed_caps = torch.stack(fixed_caps_list).to(cfg.device)

    # --- 初始化组件 ---
    my_vae = VAE(in_channels=3, latent_channels=cfg.latent_channels).to(cfg.device)
    my_text_enc = TextEncoder(vocab_size=cfg.vocab_size, embed_dim=cfg.text_embed_dim).to(cfg.device)
    my_dit = DiT(
        input_size=cfg.latent_size, 
        patch_size=2, 
        in_channels=cfg.latent_channels, 
        hidden_size=cfg.hidden_size,
        depth=1
    ).to(cfg.device)

    # 封装 FlowDiT
    model = FlowDiT(my_dit, my_vae, my_text_enc).to(cfg.device)

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=cfg.lr)

    print(f"🚀 开始训练 | 设备: {cfg.device} | Batch Size: {cfg.batch_size}")

    # --- 训练循环 ---
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for imgs, caps in progress_bar:
            imgs, caps = imgs.to(cfg.device), caps.to(cfg.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # Forward: FlowDiT 内部执行 VAE 编码 -> 加噪 -> 预测速度 -> MSE Loss
            loss = model(imgs, caps)
            
            # Backward:
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 定期采样与保存 ---
        if epoch % cfg.save_interval == 0:
            model.eval()
            print(f"\n📸 正在生成 Epoch {epoch} 的预览图...")
            
            with torch.no_grad():
                # 1. 生成图像 (基于固定文本)
                generated_imgs = model.sample(fixed_caps, steps=cfg.sample_steps)
                
                # 2. 图像逆标准化与保存
                # [B, C, H, W] -> 映射回 [0, 1] 范围
                preview: torch.Tensor = (generated_imgs * 0.5 + 0.5)
                preview = preview.clamp(0, 1)
                
                # 使用 save_image 直接保存拼图（nrow=2 表示每行 2 张图）
                img_path = os.path.join(cfg.save_dir, f"sample_epoch_{epoch}.png")
                utils.save_image(preview, img_path, nrow=2)
                
                # 3. 保存对应的文本描述
                txt_path = os.path.join(cfg.save_dir, f"sample_epoch_{epoch}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    for i in range(len(fixed_indices)):
                        ids = fixed_caps[i].tolist()
                        # 确保使用 str() 解决 Pylance 警告
                        text = "".join([str(my_vocab.get(idx, "")) for idx in ids if idx > 1])
                        f.write(f"Image {i}: {text}\n")
                
                # 4. 最后一步：保存模型权重
                # 这样做可以确保如果磁盘满了或者保存出错，你至少已经拿到了当前的生成样图
                ckpt_path = os.path.join(cfg.save_dir, f"dit_epoch_{epoch}.pth")
                torch.save(model.dit.state_dict(), ckpt_path)
                
            print(f"✅ Epoch {epoch} 记录完成：")
            print(f"   🖼️  图像: {img_path}")
            print(f"   📝  文本: {txt_path}")
            print(f"   💾  权重: {ckpt_path}")

if __name__ == '__main__':
    main()