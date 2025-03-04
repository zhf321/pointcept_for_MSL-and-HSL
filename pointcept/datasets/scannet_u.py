"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)

@DATASETS.register_module()
class ScanNetDataset_U(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        sp_path = "data/scannet_sp",
        pl_path = "pseudo_label",
        transform=None,
        test_mode=False,         
        cluster_mode=False, # 作为train 模式下面的一个开关 是否为聚类模式， 还是 train 模式
        drop_threshold=10,
        ignore_index=-1,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ScanNetDataset_U, self).__init__()
        self.data_root = data_root
        self.sp_path = sp_path
        self.pl_path = pl_path
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.cluster_mode = cluster_mode
        self.drop_threshold = drop_threshold
        self.ignore_index = ignore_index
        # self.test_cfg = test_cfg if test_mode else None

        # if test_mode:
        #     self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
        #     self.test_crop = (
        #         TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
        #     )
        #     self.post_transform = Compose(self.test_cfg.post_transform)
        #     self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):  # 'data/s3dis/Area_1/WC_1.pth'
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)] ###############
        data_sp_path = data_path.replace(self.data_root, self.sp_path)#
        if not self.cache:
            data = torch.load(data_path)
            sp_region = torch.load(data_sp_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)

        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) 
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord.astype(np.float32),
            color=color,
            segment=segment,
            region=sp_region.reshape([-1]),
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)

        if self.cluster_mode:
            data_dict["segment"][data_dict["segment"] == self.ignore_index] = -1
            # 对sp_region进行处理，去除labels = -1的， 
            # 半径滤波：以及 region中某一个类的总个数小于阈值（drop_threshold）
            sp_region = data_dict["region"]
            sp_region[data_dict["segment"] ==-1] = -1
            for q in torch.unique(sp_region):
                mask = q == sp_region
                # drop threshold 聚类的点少于该阈值就直接丢掉
                # 是因为 如果构成超点的数目过少，计算得到的特征为 Nan
                if mask.sum() < self.drop_threshold and q != -1:  
                    sp_region[mask] = -1

            valid_region = sp_region[sp_region != -1]
            unique_vals = torch.unique(valid_region)
            unique_vals.sort()
            # 使用二分查找算法，将 valid_region 中的值映射到排好序的唯一值的索引位置。
            valid_region = torch.searchsorted(unique_vals, valid_region) 

            # 将原始的 region 数组中不等于 -1 的元素替换为映射后的索引值。
            sp_region[sp_region != -1] = valid_region 
            pseudo_label = torch.ones_like(data_dict["segment"]).to(torch.int64) * -1
            data_dict["region"] = sp_region.to(data_dict["coord"].device)
        else:
            data_path = self.data_list[idx % len(self.data_list)]
            data_pl_path = data_path.replace(self.data_root, self.pl_path)#
            pseudo_label = torch.load(data_pl_path).reshape([-1])
        
        data_dict["pseudo"] = pseudo_label.to(data_dict["coord"].device)
        if data_dict["pseudo"].shape[0] != data_dict["coord"].shape[0]:
            print(f"出现错误，坐标 和 伪标签 不对应")
            print(f"scene_id: {data_dict['scene_id']}\ncoord.shape:{data_dict['coord'].shape} \n pseudo.shape: {data_dict['pseudo'].shape}")
            assert data_dict["pseudo"].shape[0] == data_dict["coord"].shape[0]
        else:
            pass
            # print(f"scene_id: {data_dict['scene_id']}\ncoord.shape:{data_dict['coord'].shape} \n pseudo.shape: {data_dict['pseudo'].shape}")
        # print(data_dict["grid_coord"].shape)
        # print(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        data_dict["segment"][data_dict["segment"] == self.ignore_index] = -1
        # 对sp_region进行处理，去除labels = -1的， 
        sp_region = data_dict["region"]
        sp_region[data_dict["segment"] ==-1] = -1

        valid_region = sp_region[sp_region != -1]
        unique_vals = torch.unique(valid_region)
        unique_vals.sort()
        # 使用二分查找算法，将 valid_region 中的值映射到排好序的唯一值的索引位置。
        valid_region = torch.searchsorted(unique_vals, valid_region) 

        # 将原始的 region 数组中不等于 -1 的元素替换为映射后的索引值。
        sp_region[sp_region != -1] = valid_region 
        data_dict["region"] = sp_region.to(data_dict["coord"].device)
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop #  % 196
