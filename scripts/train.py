import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 正在本地设备运行: {device}")

    # 初始化模型
    # # 模拟输入 (Batch, Seq_len, Dim)
    # x = torch.randn(2, 16, 512).to(device)

    # # 前向传播
    # output = model(x)
    # print(f"✅ 运行成功！输出维度: {output.shape}")


if __name__ == "__main__":
    main()
