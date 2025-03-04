import torch
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
filepath = "/home/zhaohaifeng/code/model/Pointcept-main/exp/s3dis/pretrain-msc-v1m1-0-spunet-base/model/model_last2.pth"
sys.path.append(filepath)
pretrain_model = torch.load("/home/zhaohaifeng/code/model/Pointcept-main/exp/s3dis/pretrain-msc-v1m1-0-spunet-base/model/model_last.pth")

torch.save(pretrain_model, filepath, _use_new_zipfile_serialization=False)
print("ff")