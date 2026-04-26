import torch

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np


class PokemonDataset(Dataset):

    def __init__(self, split="train[:10]", img_size=256,
                 max_token_len=96,
                 vocab=None,
                 cache_dir='./data'):
        self.raw_data = load_dataset("svjack/pokemon-blip-captions-en-zh", 
                                     split=split, cache_dir=cache_dir)
        self.img_size = img_size
        self.max_token_len = max_token_len
        self.vocab = vocab or {"<pad>": 0, "unknown": 1}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.raw_data)
    
    def _tokenize(self, text):
        """简单的分词逻辑，实际项目中建议使用 CLIPTokenizer"""
        tokens = []
        for char in text:
            tokens.append(self.vocab.get(char, 1)) 
        
        # 截断或填充
        if len(tokens) < self.max_token_len:
            tokens += [0] * (self.max_token_len - len(tokens))
        else:
            tokens = tokens[:self.max_token_len]
            
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index):
        example = self.raw_data[index]

        image = example["image"].convert("RGB")
        image_tensor = self.transform(image)

        caption = example["zh_text"]
        tokens = self._tokenize(caption)


        return image_tensor, tokens


if __name__ == "__main__":

    pokemon_ds = PokemonDataset(split="train[:10]")
    
    # 放入 DataLoader
    train_loader = DataLoader(
        pokemon_ds, 
        batch_size=2, 
        shuffle=True,
        num_workers=0 # 小规模实验建议设为 0，方便调试
    )
    
    # 测试一下输出 Shape
    imgs, caps = next(iter(train_loader))
    print(f"图像 Batch Shape: {imgs.shape}") # [2, 3, 256, 256]
    print(f"文本 Batch Shape: {caps.shape}") # [2, 77]
    
    inv_vocab = {v: k for k, v in pokemon_ds.vocab.items()}
    # 2. 开始绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in range(2):
        # --- 处理图像 ---
        img = imgs[i].numpy() # 转为 numpy [3, 256, 256]
        img = np.transpose(img, (1, 2, 0)) # 换轴 [256, 256, 3]
        
        # 逆标准化: (x * std) + mean -> (x * 0.5) + 0.5
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1) # 保证在 0-1 之间
        
        # --- 处理文本 ---
        cap_ids = caps[i].tolist()
        # 过滤掉 <pad> (ID 为 0) 并转回文字
        tokens = [inv_vocab.get(idx, "?") for idx in cap_ids if idx != 0]
        caption_text = "".join(tokens)
        
        # --- 展示 ---
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {caption_text}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()