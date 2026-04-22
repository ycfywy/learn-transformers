
import torch
from models import MultiHeadAttention

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 正在本地设备运行: {device}")

    # 初始化你自己的古法模型
    model = MultiHeadAttention(d_model=512, num_heads=8).to(device)
    
    # 模拟输入 (Batch, Seq_len, Dim)
    x = torch.randn(2, 16, 512).to(device)
    
    # 前向传播
    output = model(x)
    print(f"✅ 运行成功！输出维度: {output.shape}")

if __name__ == "__main__":
    main()