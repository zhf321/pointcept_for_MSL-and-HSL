"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    # 目前 数据都是 dict Mapping datatype
    elif isinstance(batch[0], Mapping):  # orgin_batch = batch
        orgin_batch = batch
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            # 凡是 key 中 包含 offset的 都会进行累加操作
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)

        # modify region 使得一个batch的 多个 region 连续排序，保证每个超点的单一
        # offset 和 region 一一对应 
        # offset region   key+ _ + offset  :  key + _ + region
        keys = ["", 'view1_', 'view2_']
        for key in keys:
            if f"{key}region" in batch.keys() and f"{key}offset" in batch.keys():
                region_num = 0
                region_num_list = []
                offset = batch[f"{key}offset"]
                region = batch[f"{key}region"]
                for i in range(len(offset)):
                    single_region_num = 0
                    if i == 0:
                        unit_region = region[0 : offset[i]]
                        new_region = torch.unique(unit_region)
                        single_region_num = len(new_region[new_region != -1])
                        region_num += single_region_num
                        region_num_list.append(region_num)
                    elif i != (len(offset) - 1):
                        unit_region = region[offset[i - 1] : offset[i]]
                        new_region = torch.unique(unit_region)
                        unit_region[unit_region != -1] += region_num
                        region[offset[i - 1] : offset[i]] = unit_region
                        single_region_num = len(new_region[new_region != -1])
                        region_num += single_region_num
                        region_num_list.append(region_num)
                    else:
                        unit_region = region[offset[i - 1] : offset[i]]
                        unit_region[unit_region != -1] += region_num
                        region[offset[i - 1] : offset[i]] = unit_region

                batch[f"{key}region"] = region
                batch[f"{key}region_num"] = torch.tensor(region_num_list, dtype=offset.dtype, device=offset.device)
        
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
