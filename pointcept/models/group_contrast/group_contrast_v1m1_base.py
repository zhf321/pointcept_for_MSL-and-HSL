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
from torch_scatter import scatter_mean
import torch.nn.functional as F

from pointcept.utils.sparse import sparse_sample, sparse_sample_2
from pointcept.models.group_contrast.utils import DINOHead, Head
from pointcept.models.losses.supcon_loss import SupConLoss


@MODELS.register_module("GC-v1m1")
class GroupContrast(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_in_channels,
        backbone_out_channels,

        view1_mix_prob=0,
        view2_mix_prob=0,
        matching_max_pair=8192,
        nce_t=0.4,
        proto_num=32,
        decay=0.999,
        lamda_c=0.999,
        t_q=0.1,
        t_k=0.07,
        lcon_factor=1,
        lgroup_factor=1,
    ):
        super().__init__()
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob

        # view1 和 view2 超点交集为 P
        # backbone 输出特征维度 为 D
        # 原型个数为 n 

        self.matching_max_pair = matching_max_pair
        self.proto_num = proto_num
        self.decay = decay
        self.lamda_c = lamda_c
        self.nce_t = nce_t
        # 参照DINO,  t_k 要小于 t_q
        self.t_q = t_q
        self.t_k = t_k

        self.lcon_factor = lcon_factor
        self.lgroup_factor = lgroup_factor

        self.backbone_k = build_model(backbone)
        self.backbone_q = build_model(backbone)

        # 测试一：单层 linear 层
        # self.g_k = nn.Linear(backbone_out_channels, backbone_out_channels) # (D, D)
        # self.h_k = nn.Linear(self.proto_num, backbone_out_channels)# (n, D)  .weight (D, n)

        # self.g_q = nn.Linear(backbone_out_channels, backbone_out_channels) # (D, D)
        # self.h_q = nn.Linear(self.proto_num, backbone_out_channels) # (n, D) .weight (D, n)
        # self.h_q_ = nn.Linear(self.proto_num, self.proto_num) # (n, n)

        # 测试二： mlp 3层
        # self.g_k = DINOHead(backbone_out_channels, backbone_out_channels)
        # self.h_k = DINOHead(backbone_out_channels, self.proto_num)

        # self.g_q = DINOHead(backbone_out_channels, backbone_out_channels)
        # self.h_q = DINOHead(backbone_out_channels, self.proto_num)
        # self.h_q_ = DINOHead(self.proto_num, self.proto_num)

        self.g_k = Head(backbone_out_channels, backbone_out_channels, mode="linear_relu")
        self.h_k = Head(backbone_out_channels, self.proto_num, mode="linear_relu")

        self.g_q = Head(backbone_out_channels, backbone_out_channels, mode="linear_relu")
        self.h_q = Head(backbone_out_channels, self.proto_num, mode="linear_relu")
        self.h_q_ = Head(self.proto_num, self.proto_num, mode="linear_relu")

        # proto_sk 和 proto_sq 分别是query 和 key 的原型中心
        self.proto_sk = nn.Linear(self.proto_num, backbone_out_channels, bias=False) # (n, D)
        self.proto_sq = nn.Linear(self.proto_num, backbone_out_channels, bias=False) # (n, D)

        # self.c = torch.zeros(self.proto_num, requires_grad=False).view(-1, 1)
        # self.register_buffer('c', torch.zeros(1, backbone_out_channels)) # (1, D)
        self.register_buffer('c', torch.zeros(self.proto_num, 1)) # (n, 1)

        self.apply(self._init_weights)

        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.g_q.parameters(), self.g_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  

        for param_q, param_k in zip(self.h_q.parameters(), self.h_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


        # self.mask_token = nn.Parameter(torch.zeros(1, backbone_in_channels))
        # trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.nce_criteria_none = torch.nn.CrossEntropyLoss(reduction="none")
        self.nce_criteria_mean = torch.nn.CrossEntropyLoss(reduction="mean")

        self.supcon_loss = SupConLoss(scale_by_temperature=False, is_rm_diag=False)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.no_grad()
    def _momentum_update_key_backbone(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)

        for param_q, param_k in zip(self.g_q.parameters(), self.g_k.parameters()):
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)

        for param_q, param_k in zip(self.h_q.parameters(), self.h_k.parameters()):
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)

        for param_q, param_k in zip(self.proto_sq.parameters(), self.proto_sk.parameters()):
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)
        

    def forward(self, data_dict):
        view1_origin_coord = data_dict["view1_origin_coord"]
        view1_coord = data_dict["view1_coord"]
        view1_feat = data_dict["view1_feat"]
        view1_offset = data_dict["view1_offset"].int()

        view2_origin_coord = data_dict["view2_origin_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()

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

        view1_region = data_dict['view1_region']
        view2_region = data_dict['view2_region']
        P = len(view1_region.unique())

        # SparseConv based method need grid coord
        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        # view mixing strategy
        if random.random() < self.view1_mix_prob:
            view1_data_dict["offset"] = torch.cat(
                [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0  
            ) 
        if random.random() < self.view2_mix_prob:
            # view2 —— query
            view2_data_dict["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        with torch.no_grad():
            view1_feat = self.backbone_k(view1_data_dict) # (N, D)
            # view1_feat_gk = F.linear(view1_feat, self.g_k.weight) # (N, D) * (D, D).T = (N, D)
            # view1_feat_hk = torch.matmul(view1_feat, self.h_k.weight) # (N, D) * (n, D).T = (N, n)
            # view1_feat = F.normalize(view1_feat, dim=-1, p=2)

            view1_feat_gk = self.g_k(view1_feat)
            view1_feat_hk = self.h_k(view1_feat)
            z_k = scatter_mean(F.normalize(view1_feat_gk), view1_region.view(-1, 1), dim=0) # (P, D) 
            self.c = self.c.to(z_k.device)   # (n, 1)

            z_k = F.normalize(z_k, dim=1)
            self.proto_sk.weight.copy_(F.normalize(self.proto_sk.weight, dim=0))

            K = F.softmax(
                ((torch.matmul(z_k, self.proto_sk.weight) - self.c.view(1, -1)) / self.t_k), dim=1) # (P, n)

            batch_center = torch.matmul(z_k, self.proto_sk.weight).sum(dim=0).view(-1, 1) # (n, 1)
            if dist.is_initialized():
                dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
                batch_center = batch_center / (z_k.shape[0] * dist.get_world_size())
            self.c = self.c * self.lamda_c + batch_center * (1 - self.lamda_c) # (n, 1)

            entropy_H = -torch.sum(K * torch.log(torch.clamp(K, min=1e-9)), dim=-1)  # (P,)

        view2_feat = self.backbone_q(view2_data_dict)


        view2_feat_gq = self.g_q(view2_feat)
        view2_feat_hq = self.h_q(view2_feat)
        view2_feat_hq_ = self.h_q_(view2_feat_hq)

        z_q = scatter_mean(F.normalize(view2_feat_gq), view2_region.view(-1, 1), dim=0) # (P, D)
        # (P, D)  *  (n, D).T
        z_q = F.normalize(z_q, dim=1)

        Q = F.softmax(torch.matmul(z_q, F.normalize(self.proto_sq.weight, dim=0, eps=1e-8)) / self.t_q, dim=1) # (P, n)
        log_Q = torch.log(torch.clamp(Q, min=1e-9))

        weighted_loss = torch.sum(entropy_H.unsqueeze(-1) * (K * log_Q), dim=-1) # (P,)
        # Lgroup = -torch.sum(entropy_H * weighted_loss) / torch.sum(entropy_H)
        Lgroup = -torch.sum(weighted_loss) / torch.sum(entropy_H)
        # Lgroup = -torch.sum(entropy_H.unsqueeze(-1) * (K * log_Q), dim=-1) / torch.sum(entropy_H)

        Y_seg = torch.argmax(K, dim=-1)  # (P,)

        view1_y_seg = Y_seg[view1_region]
        view2_y_seg = Y_seg[view2_region]

        view1_y_seg_bincount = view1_y_seg.bincount()#(minlength=self.proto_num)
        view2_y_seg_bincount = view2_y_seg.bincount()#(minlength=self.proto_num)
        min_y_seg_bincount = torch.min(view1_y_seg_bincount, view2_y_seg_bincount)
        
        max_samples_per_label = min_y_seg_bincount

        average_samples = self.matching_max_pair // self.proto_num
        sampling_allocation = torch.min(
            torch.full_like(max_samples_per_label, average_samples),
            max_samples_per_label
        )

        total_min_y_seg = min_y_seg_bincount.sum()  
        sampling_ratio = min_y_seg_bincount / total_min_y_seg  
        sampling_allocation = torch.clamp(sampling_allocation, max=min_y_seg_bincount)

        view1_idx_samples, view1_ptr_samples = sparse_sample_2(view1_y_seg, n_samples=sampling_allocation, return_pointers=True)
        view2_idx_samples, view2_ptr_samples = sparse_sample_2(view2_y_seg, n_samples=sampling_allocation, return_pointers=True)

        view1_samples_feats = view1_feat_hk[view1_idx_samples]
        view2_samples_feats = view2_feat_hq_[view2_idx_samples]

        view1_samples_feats = view1_samples_feats / (
            torch.norm(view1_samples_feats, p=2, dim=1, keepdim=True) + 1e-7 
        )
        view2_samples_feats = view2_samples_feats / (
            torch.norm(view2_samples_feats, p=2, dim=1, keepdim=True) + 1e-7
        )

        sim = torch.mm(view1_samples_feats, view2_samples_feats.transpose(1, 0))

        # SupCon loss--------------------------
        # positive_mask = self.create_positive_mask(sampling_allocation.cpu()).to(sim.device)
        # exp_sim_matrix =torch.exp(torch.div(sim, self.nce_t))
        # numerator = (exp_sim_matrix * positive_mask).sum(dim=1) 
        # denominator = exp_sim_matrix.sum(dim=1)
        # sim.masked_fill_(positive_mask, float('-inf'))

        view1_y_seg_samples = view1_y_seg[view1_idx_samples]
        view2_y_seg_samples = view2_y_seg[view2_idx_samples]
        confidence_weights = K[view1_y_seg_samples, :] * K[view2_y_seg_samples, :]  
        confidence_weights = torch.sum(confidence_weights, dim=1)  

        positive_mask = self.create_positive_mask(sampling_allocation.cpu()).to(sim.device)
        Lcon = self.supcon_loss(feats_sim=sim, mask=positive_mask, weight=confidence_weights)

        if torch.isnan(sim).any() or torch.isinf(sim).any():
            print("NaN or Inf found in sim matrix before Lcon calculation")
        if torch.isnan(confidence_weights).any() or torch.isinf(confidence_weights).any():
            print("NaN or Inf found in confidence_weights before Lcon calculation")


        final_loss = self.lgroup_factor * Lgroup +  self.lcon_factor * Lcon


        if dist.get_world_size() > 1:
            dist.all_reduce(Lgroup, op=dist.ReduceOp.SUM)
            dist.all_reduce(Lcon, op=dist.ReduceOp.SUM)
            final_loss = (Lgroup + Lcon) / dist.get_world_size()

        result_dict = dict(Lgroup=Lgroup / dist.get_world_size(), Lcon=Lcon / dist.get_world_size(), loss=final_loss)

        return result_dict


    @staticmethod
    def create_positive_mask(sampling_allocation):
        """
        创建正对的掩膜，保留对角线上的小方阵。
        完全使用矢量化操作，不使用 for-loop。

        :param sampling_allocation: 每个 y_seg 的采样数量，表示小方阵的边长
        :return: 正对掩膜，形状为 (n, n)，其中 n 是采样总数
        """

        n_total_samples = sampling_allocation.sum().item()

        start_indices = torch.cumsum(torch.cat([torch.tensor([0]), sampling_allocation[:-1]]), dim=0)

        mask = torch.zeros(n_total_samples, n_total_samples, dtype=torch.bool)
        reorganized_matrix = torch.zeros(n_total_samples, n_total_samples)

        for start, size in zip(start_indices, sampling_allocation):
            diag_indices = torch.arange(start, start + size)
            mask[start:start+size, start:start+size] = True
            # mask[diag_indices, diag_indices] = False

        return mask


    def get_lcon_loss(self, sampling_allocation, origin_matrix, confidence_weights):
        """
        对分组特征构建的协方差进行重组，使其满足
        每个分组的正对都位于每行最前面，负对位于每行最后面

        :param sampling_allocation: 每个 y_seg 的采样数量，表示小方阵的边长
        :param origin_matrix: 原始的协方差矩阵
        :param confidence_weights: 每一个分组的置信权重
        :return: 重组协方差矩阵，形状为 (n, n)，其中 n 是采样总数
        """

        n_total_samples = sampling_allocation.sum().item()

        start_indices = torch.cumsum(
            torch.cat([torch.tensor([0]).to(sampling_allocation.device), 
                       sampling_allocation[:-1]]), dim=0)

        reorganized_matrix = torch.zeros_like(origin_matrix)

        start_id = torch.arange(start_indices.shape[0])

        for id, start, size in zip(start_id, start_indices, sampling_allocation):
            if id == 0:
                reorganized_matrix[:size, :] = origin_matrix[:size, :]
            elif id == start_id[-1]:
                reorganized_matrix[start:start+size, 0:0+size] = origin_matrix[start:start+size, start:start+size]
                reorganized_matrix[start:start+size, size:] = origin_matrix[start:start+size, 0:start]
            else:
                reorganized_matrix[start:start+size, 0:0+size] = origin_matrix[start:start+size, start:start+size]
                reorganized_matrix[start:start+size, size:size+start] = origin_matrix[start:start+size, 0:start] 
                reorganized_matrix[start:start+size, start+size:] = origin_matrix[start:start+size, start+size:] 


        lcon_loss = torch.tensor(0.0, device=reorganized_matrix.device)

        for start, size, weight in zip(start_indices, sampling_allocation, confidence_weights):
            # neg_size = n_total_samples - pos_size
            positive_block = reorganized_matrix[start:start+size, 0:size]
            negative_block = reorganized_matrix[start:start+size, size:]

            if negative_block.numel() == 0:
                # print("Warning: negative_block is empty.")
                continue  
            
            positive_flat = positive_block.flatten().unsqueeze(1)
            negative_repeated = negative_block.repeat_interleave(size, dim=0)
            final_block = torch.cat((positive_flat, negative_repeated), dim=1)

            labels = torch.zeros(final_block.shape[0], device=final_block.device).long()
            lcon_loss = lcon_loss + weight * self.nce_criteria_mean(torch.div(final_block, self.nce_t), labels)  # 正常计算 NCE 损失
        
        return lcon_loss

