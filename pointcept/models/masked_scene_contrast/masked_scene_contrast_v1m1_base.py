"""
Masked Scene Contrast
https://arxiv.org/abs/2303.14191

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from itertools import chain
import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.nn.pool import voxel_grid

from timm.models.layers import trunc_normal_
import pointops

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size


@MODELS.register_module("MSC-v1m1")
class MaskedSceneContrast(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_in_channels,
        backbone_out_channels,
        mask_grid_size=0.1,
        mask_rate=0.4,
        view1_mix_prob=0,
        view2_mix_prob=0,
        matching_max_k=8,
        matching_max_radius=0.03,
        matching_max_pair=8192,
        nce_t=0.4,
        contrast_weight=1,
        reconstruct_weight=1,
        reconstruct_color=True,
        reconstruct_normal=True,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.mask_grid_size = mask_grid_size
        self.mask_rate = mask_rate
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob
        self.matching_max_k = matching_max_k
        self.matching_max_radius = matching_max_radius
        self.matching_max_pair = matching_max_pair
        self.nce_t = nce_t
        self.contrast_weight = contrast_weight
        self.reconstruct_weight = reconstruct_weight
        self.reconstruct_color = reconstruct_color
        self.reconstruct_normal = reconstruct_normal

        self.mask_token = nn.Parameter(torch.zeros(1, backbone_in_channels))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.color_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_color else None
        )
        self.normal_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_normal else None
        )
        self.nce_criteria = torch.nn.CrossEntropyLoss(reduction="mean")

    @torch.no_grad()
    def generate_cross_masks(
        self, view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
    ):
        # union origin coord
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        view1_origin_coord_split = view1_origin_coord.split(list(view1_batch_count))
        view2_origin_coord_split = view2_origin_coord.split(list(view2_batch_count))
        union_origin_coord = torch.cat(
            list(
                chain.from_iterable(
                    zip(view1_origin_coord_split, view2_origin_coord_split)
                )
            )
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        # grid partition
        mask_patch_coord = union_origin_coord.div(self.mask_grid_size)
        mask_patch_grid_coord = torch.floor(mask_patch_coord)
        # mask_patch_cluster : 记录每一个点所属体素的索引
        mask_patch_cluster = voxel_grid(
            pos=mask_patch_grid_coord, size=1, batch=union_batch, start=0
        )
        unique, cluster, counts = torch.unique(
            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
        )
        # 一共有patch_num个体素不为空
        patch_num = unique.shape[0]
        # patch_max_point：单个体素中最多点的个数
        patch_max_point = counts.max().item()
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        # patch2point_mask:[patch_num, patch_max_point]:每一行 代表一个体素，
        # 体素中有多少个点，假设为 n , 则，[0, n-1] 为True, [n, patch_max_point]为 False
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point).cuda().unsqueeze(0), counts.unsqueeze(-1)
        )
        sorted_cluster_value, sorted_cluster_indices = torch.sort(cluster)
        # 二维张量使用shape相同的二维布尔索引，返回的是一维张量
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        # generate cross masks
        assert self.mask_rate <= 0.5
        patch_mask = torch.zeros(patch_num, device=union_origin_coord.device).int()
        rand_perm = torch.randperm(patch_num)
        mask_patch_num = int(patch_num * self.mask_rate) # 要mask 多少个带有点云的体素

        # mask1 tag with 1, mask2 tag with 2
        patch_mask[rand_perm[0:mask_patch_num]] = 1
        patch_mask[rand_perm[mask_patch_num : mask_patch_num * 2]] = 2
        point_mask = torch.zeros(
            union_origin_coord.shape[0], device=union_origin_coord.device
        ).int()
        # patch2point_map[patch_mask == 1].shape 和 patch2point_mask[patch_mask == 1].shape 是相同的
        # 就又变成了二维张量和二维索引shape相同的情况，得到的是一维索引
        # 表达式的含义是从 patch2point_map 中选择那些
        # 在 patch_mask 为1且对应位置在 patch2point_mask 也为 True 的元素
        point_mask[
            patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]] 
        ] = 1
        point_mask[
            patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]
        ] = 2

        # separate mask to view1 and view2
        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)],
                    dim=-1,
                ).flatten()
            )
        )
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2
        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def match_contrastive_pair(
        self, view1_coord, view1_offset, view2_coord, view2_offset, max_k, max_radius
    ):  
        # 执行 KNN 查询，找到每个新坐标点最近的 nsample 个邻居，并返回它们的索引和距离。
        # index.shape:(view1_coord.shape[0], max_k) distance.shape:(view1_coord.shape[0], max_k)
        index, distance = pointops.knn_query(  
            max_k,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        # index 经过索引之后为 二维张量，包含每个点及其最近邻点序号 第0维为每个点序号，第1维为最近邻点序号
        # 待索引数据shape(a, b, c)  索引 shape(a, b), 那么得到数据即为 (x, c) x = 索引.sum()
        index = torch.cat(   
            [
                torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                .view(-1, 1, 1)  
                .expand(-1, max_k, 1),
                index.view(-1, max_k, 1),
            ],
            dim=-1,
        )[distance.squeeze(-1) < max_radius] 
        # unique.size 表示一共有这么多的点有符合条件的近邻点
        # index[:, 0] 这个0代表着 每个点的序号这列
        unique, count = index[:, 0].unique(return_counts=True)  
        # 相当于从每个点的knn查询得到的符合条件的点中，随机选择一个
        select = ( 
            # 最后面又减去了 1 是因为 torch.randint(cout.max())  最大值是取到 cout.max() - 1
            # torch.randint() low 默认为 0, 取到 0 时， 对于最后一个数据而言，就越界了，故 -1
            # 以count[0]为例： %count 范围为： [0, count[0] - 1]; 
            # count[0] - (%count) 范围：[1, count[0]]; 再 -1 范围为：[0, count[0] - 1]
            torch.cumsum(count, dim=0)
            - torch.randint(count.max(), count.shape, device=count.device) % count
            - 1   
        ) 
        # 经过select 之后的index 为二维，第一维：具有近邻点的点序号，唯一，
        # 第二维，点序号对应随机选中的近邻点序号，不唯一，即两个点的选中的近邻点可能是同一个
        index = index[select] 
        if index.shape[0] > self.matching_max_pair:
            # 先随机洗牌，然后截取前 max_pair个，然后index[]再应用该索引
            index = index[torch.randperm(index.shape[0])[: self.matching_max_pair]] 
        return index

    def compute_contrastive_loss(
        self, view1_feat, view1_offset, view2_feat, view2_offset, match_index
    ):
        assert view1_offset.shape == view2_offset.shape

        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]
        view1_feat = view1_feat / (
            torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7   # 向量特征归一化
        )
        view2_feat = view2_feat / (
            torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        sim = torch.mm(view1_feat, view2_feat.transpose(1, 0))   # 论文中公式（1）

        with torch.no_grad():
            pos_sim = torch.diagonal(sim).mean()
            # 因为sim.mean(dim=-1).mean() 这个相当于算的时候对正对 除以两次 8192 ，
            # 所以后面也要再除一下
            neg_sim = sim.mean(dim=-1).mean() - pos_sim / match_index.shape[0]
        # sim 正方形矩阵的对角线为 正对，所以 label 刚好就是对角线的下标
        labels = torch.arange(sim.shape[0], device=view1_feat.device).long()
        loss = self.nce_criteria(torch.div(sim, self.nce_t), labels)

        if get_world_size() > 1:
            dist.all_reduce(loss)
            dist.all_reduce(pos_sim)
            dist.all_reduce(neg_sim)
        return (
            loss / get_world_size(),
            pos_sim / get_world_size(),
            neg_sim / get_world_size(),
        )

    def forward(self, data_dict):
        view1_origin_coord = data_dict["view1_origin_coord"]
        view1_coord = data_dict["view1_coord"]
        view1_feat = data_dict["view1_feat"]
        view1_offset = data_dict["view1_offset"].int()

        view2_origin_coord = data_dict["view2_origin_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()

        # mask generation by union original coord (without spatial augmentation)
        view1_point_mask, view2_point_mask = self.generate_cross_masks(
            view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
        )

        view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
        view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
        view1_feat = view1_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

        view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
        view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
        view2_feat = view2_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

        view1_data_dict = dict(
            origin_coord=view1_origin_coord,
            coord=view1_coord,
            feat=view1_feat,
            offset=view1_offset,
        )
        view2_data_dict = dict(
            origin_coord=view2_origin_coord,
            coord=view2_coord,
            feat=view2_feat,
            offset=view2_offset,
        )

        # SparseConv based method need grid coord
        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        # view mixing strategy
        if random.random() < self.view1_mix_prob:
            # view1_offset[-1].shape 为 torch.Size([]) , 所以需要扩展下：变成：torch.Size([1])
            # 隔两个取一个，就代表两个变成了一个，mix
            view1_data_dict["offset"] = torch.cat(
                [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0  
            ) 
        if random.random() < self.view2_mix_prob:
            view2_data_dict["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        view1_feat = self.backbone(view1_data_dict)
        view2_feat = self.backbone(view2_data_dict)
        match_index = self.match_contrastive_pair(
            view1_origin_coord,
            view1_offset,
            view2_origin_coord,
            view2_offset,
            max_k=self.matching_max_k,
            max_radius=self.matching_max_radius,
        )
        nce_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
            view1_feat, view1_offset, view2_feat, view2_offset, match_index
        )
        loss = nce_loss * self.contrast_weight
        result_dict = dict(nce_loss=nce_loss, pos_sim=pos_sim, neg_sim=neg_sim)

        if self.color_head is not None:
            assert "view1_color" in data_dict.keys()
            assert "view2_color" in data_dict.keys()
            view1_color = data_dict["view1_color"]
            view2_color = data_dict["view2_color"]
            view1_color_pred = self.color_head(view1_feat[view1_point_mask])
            view2_color_pred = self.color_head(view2_feat[view2_point_mask])
            color_loss = (
                torch.sum((view1_color_pred - view1_color[view1_point_mask]) ** 2)
                + torch.sum((view2_color_pred - view2_color[view2_point_mask]) ** 2)
            ) / (view1_color_pred.shape[0] + view2_color_pred.shape[0])
            loss = loss + color_loss * self.reconstruct_weight
            result_dict["color_loss"] = color_loss

        if self.normal_head is not None:
            assert "view1_normal" in data_dict.keys()
            assert "view2_normal" in data_dict.keys()
            view1_normal = data_dict["view1_normal"]
            view2_normal = data_dict["view2_normal"]
            view1_normal_pred = self.normal_head(view1_feat[view1_point_mask])
            view2_normal_pred = self.normal_head(view2_feat[view2_point_mask])

            view1_normal_pred = view1_normal_pred / (
                torch.norm(view1_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            view2_normal_pred = view2_normal_pred / (
                torch.norm(view2_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            normal_loss = (
                torch.sum(view1_normal_pred * view1_normal[view1_point_mask])
                + torch.sum(view2_normal_pred * view2_normal[view2_point_mask])
            ) / (view1_normal_pred.shape[0] + view2_normal_pred.shape[0])
            loss = loss + normal_loss * self.reconstruct_weight
            result_dict["normal_loss"] = normal_loss

        result_dict["loss"] = loss
        return result_dict

# a = MaskedSceneContrast()
# print("test")