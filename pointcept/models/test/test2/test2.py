import torch
model_path = "/home/zhaohaifeng/code/model/Pointcept-main/exp/s3dis/0-semseg-minkunet34c-0-base-u-test/model/model_best.pth"
a = torch.load(model_path)

print("ff")