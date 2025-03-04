import torch

data_path = "/home/zhaohaifeng/code/model/Pointcept-main/exp/s3dis/0-semseg-minkunet14c-0-base-u-b-more_clu_freq_normal_mix-test/pseudo_label/Area_1/office_30.pth"

data = torch.load(data_path)
print(data.shape)

print("ff")