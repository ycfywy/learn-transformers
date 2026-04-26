import torch

# 原始向量：3个元素 (时间步)
t = torch.tensor([1, 5, 10]) 
print(f"原始 t 形状: {t.shape}") # torch.Size([3])

# 情况 A：t[:, None] -> 在列的位置加维度
t_col = t[:, None]
print(f"t[:, None] 形状: {t_col.shape}") # torch.Size([3, 1])
# 值变成了: [[1], [5], [10]]

# 情况 B：t[None, :] -> 在行的位置加维度
t_row = t[None, :]
print(f"t[None, :] 形状: {t_row.shape}") # torch.Size([1, 3])
# 值变成了: [[1, 5, 10]]


# 假设我们有两组特征，形状都是 (3, 2)
# 3 代表 batch_size，2 代表特征维度
a = torch.tensor([[1, 1], [2, 2], [3, 3]])
b = torch.tensor([[4, 4], [5, 5], [6, 6]])

# dim=0: 纵向拼接（增加行数）
res_dim0 = torch.cat([a, b], dim=0)
print(f"dim=0 拼接后形状: {res_dim0.shape}") # torch.Size([6, 2])
# 结果：
# [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]

# dim=-1 (或 dim=1): 横向拼接（增加列数/特征长度）
res_dim1 = torch.cat([a, b], dim=-1)
print(f"dim=-1 拼接后形状: {res_dim1.shape}") # torch.Size([3, 4])
# 结果：
# [[1, 1, 4, 4], [2, 2, 5, 5], [3, 3, 6, 6]]


import torch
import torch.nn as nn

# 定义一个输入 4 通道，输出 10 通道，卷积核 3x3 的层
conv = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3)

# 查看权重（weight）的形状
print(conv.weight.shape) 
# 输出: torch.Size([10, 4, 3, 3])


import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

model = MyModel()
print(list(model.parameters())) # 结果是 []，参数“丢失”了