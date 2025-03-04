import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 获取模型的所有参数
parameters = model.parameters()

# 打印每个参数的形状
for param in parameters:
    print(param.shape)

# from pointcept.models.test.test1 import aaa

# print(aaa)