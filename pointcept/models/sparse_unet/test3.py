import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseSequential

# 假设你的初始 x 如下
x = SparseConvTensor(
    features=torch.rand(19366, 6),
    indices=torch.randint(0, 187, (19366, 4)),
    spatial_shape=[187, 187, 126]
)

# 定义卷积层
conv_layer = SparseSequential(
    SubMConv3d(6, 32, kernel_size=[5, 5, 5], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo="ConvAlgo.Native"),
    nn.BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
    nn.ReLU()
)

# 进行稀疏卷积
x = conv_layer(x)

# 输出结果
print(f"x.features.shape: {x.features.shape}    x.indices.shape: {x.indices.shape}     x.spatial_shape: {x.spatial_shape}")
