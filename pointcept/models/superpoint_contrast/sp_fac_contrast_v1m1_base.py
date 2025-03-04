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


@MODELS.register_module("SFAC-v1m1")
class SpFAContrast(nn.Module):
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

        self.g_k = Head(backbone_out_channels, backbone_out_channels, mode="linear_bn_relu")
        self.h_k = Head(backbone_out_channels, self.proto_num, mode="linear_bn_relu")

        self.g_q = Head(backbone_out_channels, backbone_out_channels, mode="linear_bn_relu")
        self.h_q = Head(backbone_out_channels, self.proto_num, mode="linear_bn_relu")
        self.h_q_ = Head(self.proto_num, self.proto_num, mode="linear_bn_relu")
        # self.c = torch.zeros(self.proto_num, requires_grad=False).view(-1, 1)
        self.register_buffer('c', torch.zeros(self.proto_num, 1))


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

        self.nce_criteria_none = torch.nn.CrossEntropyLoss(reduction="none")
        self.nce_criteria_mean = torch.nn.CrossEntropyLoss(reduction="mean")

        self.supcon_loss = SupConLoss(temperature=0.5, scale_by_temperature=False, is_rm_diag=False)


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
            # param_k.data = param_k.data * self.decay + param_q.data * (1. - self.decay)
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)

        for param_q, param_k in zip(self.g_q.parameters(), self.g_k.parameters()):
            param_k.data.mul_(self.decay).add_((1. - self.decay) * param_q.data)

        for param_q, param_k in zip(self.h_q.parameters(), self.h_k.parameters()):
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
            view2_data_dict["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        with torch.no_grad():
            view1_feat = self.backbone_k(view1_data_dict) # (N, D)
            view1_feat_gk = self.g_k(view1_feat)

        view2_feat = self.backbone_q(view2_data_dict) # (N, D)
        view2_feat_gq = self.g_q(view2_feat)

        view1_sp_wall_mask = data_dict["view1_sp_wall_mask"]
        view1_sp_floor_ceil_mask = data_dict["view1_sp_floor_ceil_mask"]
        view1_sp_background_mask = view1_sp_wall_mask | view1_sp_floor_ceil_mask


        view1_region = data_dict['view1_region']
        view2_region = data_dict['view2_region']
        view1_sp_seg_bincount = view1_region.bincount()
        view2_sp_seg_bincount = view2_region.bincount()
       
        min_y_seg_bincount = torch.min(view1_sp_seg_bincount, view2_sp_seg_bincount)
        small_sp_mask = min_y_seg_bincount < 30 # 
        view2_anchor_feats = scatter_mean(view2_feat, view2_region, dim=0)
        view2_anchor_feats = view2_anchor_feats[~small_sp_mask]

        min_y_seg_bincount[small_sp_mask] = 0
       
        each_sp_sampling_size = int(self.matching_max_pair / (~small_sp_mask).sum())
        min_y_seg_bincount[~small_sp_mask] = 30 if each_sp_sampling_size >= 30 else each_sp_sampling_size

        sampling_allocation = min_y_seg_bincount
        view1_idx_samples, view1_ptr_samples = sparse_sample_2(view1_region, n_samples=sampling_allocation, return_pointers=True)
        view1_samples_feats = view1_feat[view1_idx_samples]
       
        view1_samples_feats = view1_samples_feats / (
            torch.norm(view1_samples_feats, p=2, dim=1, keepdim=True) + 1e-7)
        view2_anchor_feats = view2_anchor_feats / (
            torch.norm(view2_anchor_feats, p=2, dim=1, keepdim=True) + 1e-7)
        sim = torch.mm(view2_anchor_feats, view1_samples_feats.transpose(1, 0)) #(n, N)
     
        positive_mask = self.create_positive_mask_2(sim).to(sim.device)
        negatives_mask_f = self.create_negative_mask_from_positive_mask(positive_mask=positive_mask, sample_count=30)
        Lcon = self.supcon_loss(feats_sim=sim, mask=positive_mask, 
                                is_positive_in_denom=True, negatives_mask=negatives_mask_f)

        view2_anchor_gq_feats = scatter_mean(view2_feat_gq, view2_region, dim=0)
        view2_anchor_gq_feats = view2_anchor_gq_feats[~small_sp_mask]

        view2_anchor_gq_feats = view2_anchor_gq_feats / (
            torch.norm(view2_anchor_gq_feats, p=2, dim=1, keepdim=True) + 1e-7)
        view1_samples_gk_feats = view1_feat_gk[view1_idx_samples]
        view1_samples_gk_feats = view1_samples_gk_feats / (
            torch.norm(view1_samples_gk_feats, p=2, dim=1, keepdim=True) + 1e-7)
        
        sim_2 = torch.mm(view2_anchor_gq_feats, view1_samples_gk_feats.transpose(1, 0)) #(n, N)
       
        positive_mask_2 = self.create_positive_mask_3(sim_2).to(sim_2.device)
       
        view1_sp_background_mask = view1_sp_background_mask[~small_sp_mask]
        
        if view1_sp_background_mask.sum() == 0 :
            print(f"view1_sp_background_mask.sum(): {view1_sp_background_mask.sum()}")
            Lcon_2 = 0
        else:
            negatives_mask = self.create_negative_mask(sim_2, view1_sp_background_mask, sample_count=30).to(sim_2.device)
            Lcon_2 = self.supcon_loss(feats_sim=sim_2, mask=positive_mask_2, is_positive_in_denom=True, negatives_mask=negatives_mask)
            

        if dist.get_world_size() > 1:
            dist.all_reduce(Lcon, op=dist.ReduceOp.SUM)
            if Lcon_2 != 0:
                dist.all_reduce(Lcon_2, op=dist.ReduceOp.SUM)
            dist.all_reduce(Lcon_2, op=dist.ReduceOp.SUM)
            L_total = Lcon + Lcon_2#
            final_loss = (L_total) / dist.get_world_size()


        result_dict = dict(L_fore=Lcon / dist.get_world_size(), L_fore_back=Lcon_2 / dist.get_world_size(), loss=final_loss)

        return result_dict
    

    @staticmethod
    def create_positive_mask_2(sim):
        n, N = sim.shape
        k = N // n  
        mask = torch.zeros((n, N), dtype=torch.bool)

        for i in range(n):
            start_idx = i * k
            end_idx = (i + 1) * k
            mask[i, start_idx:end_idx] = True
        
        return mask


    @staticmethod
    def create_positive_mask_3(sim):
        
        n, N = sim.shape
        k = N // n  
        mask = torch.zeros((n, N), dtype=torch.bool)
        top_k_values, top_k_indices = torch.topk(sim, k, dim=1)

        for i in range(n):
            mask[i, top_k_indices[i]] = True

        return mask


    @staticmethod
    def create_negative_mask_from_positive_mask(positive_mask, sample_count):
        negative_mask = ~positive_mask

        n, N = negative_mask.shape
        for i in range(n):
            negative_indices = torch.nonzero(negative_mask[i]).squeeze()  
            if negative_indices.numel() > 1: 
                sampled_indices = negative_indices[torch.linspace(0, len(negative_indices) - 1, steps=sample_count).long()]
                negative_mask[i] = False  
                negative_mask[i, sampled_indices] = True 
            else:
                negative_mask[i] = False

        return negative_mask

    @staticmethod
    def create_negative_mask(sim, sp_background_mask, sample_count=30):
        
        n, N = sim.shape
        k = N // n  
        mask = torch.zeros((n, N), dtype=torch.bool)

        for i in range(n):
            start_idx = i * k
            end_idx = (i + 1) * k
            if sp_background_mask[i]:  
                mask[i, start_idx:end_idx] = False
                mask[i, ~sp_background_mask.repeat_interleave(k)] = True
            else:  
                mask[i, start_idx:end_idx] = False
                mask[i, sp_background_mask.repeat_interleave(k)] = True

            negative_indices = torch.nonzero(mask[i]).squeeze()  
            if negative_indices.numel() > sample_count:  
                sampled_indices = negative_indices[torch.linspace(0, len(negative_indices) - 1, steps=sample_count).long()]
                mask[i] = False  
                mask[i, sampled_indices] = True  
            else:
                mask[i] = False

        return mask



    @staticmethod
    def create_positive_mask(sampling_allocation):
        n_total_samples = sampling_allocation.sum().item()
        start_indices = torch.cumsum(torch.cat([torch.tensor([0]), sampling_allocation[:-1]]), dim=0)
        mask = torch.zeros(n_total_samples, n_total_samples, dtype=torch.bool)
        reorganized_matrix = torch.zeros(n_total_samples, n_total_samples)

        for start, size in zip(start_indices, sampling_allocation):
            diag_indices = torch.arange(start, start + size)
            mask[start:start+size, start:start+size] = True

        return mask


    def get_lcon_loss(self, sampling_allocation, origin_matrix, confidence_weights=None):
        
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
            # print(f"id: {id}")
            # print(f"reorganized_matrix: \n{reorganized_matrix}")

        lcon_loss = torch.tensor(0.0, device=reorganized_matrix.device)

        for start, size in zip(start_indices, sampling_allocation):
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
            lcon_loss = lcon_loss + self.nce_criteria_mean(torch.div(final_block, self.nce_t), labels)  
           
        return lcon_loss
