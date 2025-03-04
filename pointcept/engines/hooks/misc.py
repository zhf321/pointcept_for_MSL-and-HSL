"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from sklearn.cluster import KMeans

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize, get_world_size
from pointcept.utils.cache import shared_dict

import pointcept.utils.comm as comm
from pointcept.engines.test import TESTERS

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        # if self.trainer.cfg.resume:  ##########
        self._remain_iter = (self.trainer.max_epoch - self.trainer.start_epoch) * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self, log_interval=1):
        self.curr_iter = 0
        self.model_output_keys = []
        self.log_interval = log_interval

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        # MSC pretrain do not have offset information. Comment the code for support MSC
        # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
        #        "Scan {batch_size} ({points_num}) ".format(
        #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
        #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
        #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
        #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
        # )
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)

        # if self.curr_iter % len(self.trainer.train_loader) == self.log_interval:
        if (self.curr_iter % len(self.trainer.train_loader)) % self.log_interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])

        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None,
                    "classifier" : self.trainer.classifier.state_dict()
                    if hasattr(self.trainer, "classifier")
                    else None,
                    "current_growsp": self.trainer.current_growsp
                    if hasattr(self.trainer, "current_growsp")
                    else None,
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointSaver_U(HookBase):
    def __init__(self, save_freq=1):
        self.save_freq = save_freq  

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate and (self.trainer.epoch + 1) % self.save_freq == 0:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

                filename = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_last.pth"
                )
                self.trainer.logger.info("Saving checkpoint to: " + filename)
                torch.save(
                    {
                        "epoch": self.trainer.epoch + 1,
                        "state_dict": self.trainer.model.state_dict(),
                        "optimizer": self.trainer.optimizer.state_dict(),
                        "scheduler": self.trainer.scheduler.state_dict(),
                        "scaler": self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None,
                        "classifier" : self.trainer.classifier.state_dict()
                        if hasattr(self.trainer, "classifier")
                        else None,
                        "current_growsp": self.trainer.current_growsp
                        if hasattr(self.trainer, "current_growsp")
                        else None,
                        "best_metric_value": self.trainer.best_metric_value,
                    },
                    filename + ".tmp",
                )
                os.replace(filename + ".tmp", filename)

            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )

@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    if comm.get_world_size() >= 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
                if hasattr(self.trainer, "current_growsp"):
                    self.trainer.current_growsp = checkpoint["current_growsp"]
                if hasattr(self.trainer, "classifier"):
                    # if self.trainer.current_growsp is not None:
                    #     out_features = self.trainer.current_growsp 
                    # else:
                    #     out_features = self.trainer.cluster_cfg.primitive_num
                    # self.trainer.classifier = torch.nn.Linear(in_features=self.trainer.cluster_cfg.feats_dim,
                    #                                           out_features=out_features,
                    #                                           bias=False)
                    self.trainer.classifier = torch.nn.Linear(in_features=self.trainer.cluster_cfg.feats_dim,
                                                              out_features=self.trainer.cluster_cfg.primitive_num,
                                                              bias=False)
                    self.trainer.classifier.load_state_dict(checkpoint["classifier"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path)
            state_dict = checkpoint["state_dict"]
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()


# 对超点的特征进行聚类，用于原型中心的初始化
# @HOOKS.register_module()
# class Cluster(HookBase):

    # def get_sp_hf_nn_feature(self, dataloader, level=1, model=None, is_spconv=False):
    #     """获取超点的手工特征(level-超点:pfh + size +  rgb) 以及神经网络特征
    #         用于聚类
    #     返回值：
    #         sp_feats_list: 超点特征list,  list[index] 与 dataset 一一对应
    #                     sp_feats_list 里面有 202个 值, 每个值 用于存储该room 的超点特征，按照超点的顺序
    #         sp_num_list : 超点数量, 同样 list[index] 与 dataset 一一对应
    #             更确切地说, 是dataset 里面的 cloud_ids[index] 一一对应
    #             也是有 202个值，每个值表示该room 里面超点的个数
    #     """
    #     # level 应该是 4
    #     sp_feats_hf_nn_ordered_list = [[] for _ in range(len(dataloader.dataset))]

    #     with torch.no_grad():
    #         for i, batch in enumerate(dataloader):
    #             assert isinstance(batch, list)
    #             batch = self._batch_to_device(batch, self.device)
    #             batch = self.trainer.datamodule.naglist2batch(batch)
    #             # 用于神经网络特征提取
    #             # transfer_batch = self.trainer.datamodule.on_after_batch_transfer(batch, i)
    #             transfer_batch = batch
    #             transfer_batch_index = transfer_batch[0].norm_index(mode='graph').view(-1, 1)

    #             # 用于神经网络计算的 features xyz_rgb_normal
    #             features = torch.cat((batch[0].rgb_3, batch[0].pos_trans), dim=1)
    #             features = features.float()
    #             grid_coords = transfer_batch[0].grid_coords
    #             # grid_coords = grid_coords - grid_coords.min(dim=0).values.round().int()
    #             transfer_batch_coords = torch.cat([transfer_batch_index, grid_coords], dim=1).float()
                
    #             # 基于 spconv 获取逐点神经网络特征
    #             if is_spconv:
    #                 batch_spconv = transfer_batch_coords[:, 0].int()
    #                 coords_batch_shape = torch.max(transfer_batch_coords[:,1:4], dim=0).values 
    #                 # print(f"coords_batch_shape: {coords_batch_shape}")
    #                 sparse_shape = torch.add(coords_batch_shape, 96).tolist()
    #                 in_field = spconv.SparseConvTensor(
    #                     features=features.cuda(),
    #                     indices=transfer_batch_coords.int().cuda(),
    #                     spatial_shape=sparse_shape,
    #                     batch_size=batch_spconv[-1].tolist() + 1,
    #                 )
    #             else:
    #                 in_field = ME.TensorField(features.cuda(), transfer_batch_coords.cuda(), device=transfer_batch.device)
    #             feats_nn = model(in_field)

    #             # TODO: 在这里需不需 加入 l2-normalize
    #             feats_nn = F.normalize(feats_nn, dim=1)

    #             # feats_nn = F.normalize(feats_nn, dim=1) # 是否对生成的 点级别特征直接 normalize
    #             super_index = transfer_batch.get_super_index(high=level)
    #             feats_nn_sp = scatter_mean(feats_nn, super_index.to(feats_nn.device), dim=0)
    #             feats_nn_sp = F.normalize(feats_nn_sp, dim=1)
        
    #             # 用于手工特征提取
    #             feats_pfh = batch[level].pfh  # 10
    #             feats_size = batch[level].log_size  # 1
    #             feats_rgb = batch[level].rgb # 3
    #             feats_a_lwh = batch[level].a_lwh  # 3
    #             feats_o_lwh = batch[level].o_lwh  # 3
    #             feats_z_lwh = batch[level].z_lwh  # 3
    #             feats_z = batch[level].pos[:, -1].view(-1, 1)
    #             feats_min_z = batch[level].min_pos[:, -1].view(-1, 1)
    #             feats_max_z = batch[level].max_pos[:, -1].view(-1, 1)
    #             feats_y_hist = batch[level].y
    #             feats_normal = batch[level].normal

    #             # print(f"feats_y_hist: {feats_y_hist.shape}")

    #             feats_sp_hf_nn = torch.cat([feats_pfh, feats_size, feats_rgb, 
    #                                         feats_a_lwh, feats_o_lwh, feats_z_lwh, 
    #                                         feats_z, feats_min_z, feats_max_z, feats_y_hist,
    #                                         feats_nn_sp, feats_normal], dim=1)

    #             index_batch = batch.index_batch
    #             # print(f"index_batch: {index_batch}")
    #             transfer_index_batch = transfer_batch.index_batch
    #             # print(f"transfer_index_batch: {transfer_index_batch}")
    #             batch_pointers = indices_to_pointers(batch[level].batch)[0]
                
    #             for i, index in enumerate(index_batch):
    #                 sp_feats_hf_nn_ordered_list[index].append(feats_sp_hf_nn[batch_pointers[i] : batch_pointers[i + 1]])

    #     # 将嵌套list 改为非嵌套list， 去除掉里面一层list
    #     # sp_feats_hf_list 和 sp_num_list 两个分别代表： dataset[index] 数据对应 sp_feats_list[index]的超点手工特征
    #     # 以及 sp_num_list[index] 超点的数量
    #     sp_feats_hf_nn_list = []
    #     sp_num_list = []
    #     for i in range(len(sp_feats_hf_nn_ordered_list)):
    #         sp_feats = sp_feats_hf_nn_ordered_list[i][0]
    #         sp_num = sp_feats.shape[0]
    #         sp_feats_hf_nn_list.append(sp_feats)
    #         sp_num_list.append(sp_num)

    #     return sp_feats_hf_nn_list, sp_num_list



@HOOKS.register_module()
class ClusterClassifier(HookBase):
    def __init__(self, cluster_freq = 1, mix_mode=False):
        # self.current_growsp = None
        self.if_Growing = False
        self.cluster_freq = cluster_freq
        self.mix_mode = mix_mode

    def before_epoch(self):
        if self.trainer.epoch == self.trainer.cluster_cfg.start_grow_epoch:
            self.if_Growing = True
            # 更新 优化器和 调度器
            self.trainer.optimizer = self.trainer.build_optimizer()
            total_epoch = self.trainer.max_epoch - self.trainer.cluster_cfg.start_grow_epoch
            self.trainer.scheduler = self.trainer.build_scheduler(total_epoch)

        if self.trainer.epoch % self.cluster_freq == 0:
            self.trainer.classifier = self.cluster()

    def cluster(self):
        time_start = time.time()

        if self.if_Growing:
            growth_rate = (self.trainer.epoch - self.trainer.cluster_cfg.start_grow_epoch) \
                        / (self.trainer.max_epoch - self.trainer.cluster_cfg.start_grow_epoch)
            self.trainer.current_growsp =  int(
                self.trainer.cluster_cfg.grow_start - \
                    growth_rate * (self.trainer.cluster_cfg.grow_start - self.trainer.cluster_cfg.grow_end)
                )

            if self.trainer.current_growsp < self.trainer.cluster_cfg.grow_end:
                self.trainer.current_growsp = self.trainer.cluster_cfg.grow_end
            self.trainer.logger.info('Epoch: {}, Superpoints Grow to {}'.format(self.trainer.epoch, self.trainer.current_growsp))
        feats, labels, sp_index, context = ClusterClassifier.get_sp_feature(
            self.trainer.cluster_cfg,
            self.trainer.cluster_loader, 
            self.trainer.model.backbone, 
            self.trainer.current_growsp,
            self.mix_mode)
        time_get_sp_feature = time.time()
        print(f"get_sp_feature spend time: {time_get_sp_feature - time_start}")
        # will do Kmeans with geometric distance  所有train的203个room的所有超点的特征 # torch.Size([32246, 141])
        # 把所有train_room的超点直接聚成300基元类
        sp_feats = torch.cat(feats, dim=0)  # all_neural_region 的特征 整合到一块了 也就是 超点的顺序 以此累加了
        primitive_labels = KMeans(n_clusters=self.trainer.cluster_cfg.primitive_num, 
                                n_init=5, 
                                random_state=0).fit_predict(sp_feats.cpu().numpy())
        primitive_labels = torch.from_numpy(primitive_labels).to(sp_index[0].device)
        # drop geometric feature 只保留 由model 提取的 128维特征，去掉rgb 和 pfh特征 # torch.Size([32246, 128])
        sp_feats = sp_feats[:,0:self.trainer.cluster_cfg.feats_dim]
        time_kmeans = time.time()
        print(f"kmeans spend time: {time_kmeans - time_get_sp_feature}")

        '''Compute Primitive Centers'''
        primitive_centers = torch.zeros((self.trainer.cluster_cfg.primitive_num, self.trainer.cluster_cfg.feats_dim))
        for cluster_idx in range(self.trainer.cluster_cfg.primitive_num):
            indices = primitive_labels == cluster_idx
            cluster_avg = sp_feats[indices].mean(0, keepdims=True)
            primitive_centers[cluster_idx] = cluster_avg
        primitive_centers = F.normalize(primitive_centers, dim=1)
        classifier = ClusterClassifier.get_fixclassifier(in_channel=self.trainer.cluster_cfg.feats_dim, 
                                    centroids_num=self.trainer.cluster_cfg.primitive_num, 
                                    centroids=primitive_centers)
        
        '''Compute and Save Pseudo Labels'''
        # primitive_labels 所有train_room的超点经过 KMeans 聚类之后生成的 label 
        all_pseudo, all_gt, all_pseudo_gt = ClusterClassifier.get_pseudo(self.trainer.cluster_loader, 
                                                                         context, 
                                                                         primitive_labels, 
                                                                         sp_index)
        print(f"计算并保存pseudo_label spend time : {time.time() - time_kmeans}")
        self.trainer.logger.info('labelled points ratio %.2f clustering time: %.2fs', 
                    (all_pseudo!=-1).sum()/all_pseudo.shape[0], time.time() - time_start)
        
        '''Check Superpoint/Primitive Acc in Training'''
        sem_num = self.trainer.cfg.data.num_classes
        mask = (all_pseudo_gt!=-1)
        # hungarian matching
        # 计算超点 对应的 真值
        # sem_num * all_gt + all_pseudo_gt 这样做 类似与 sem_num 进制
        histogram = torch.bincount(sem_num* all_gt.to(torch.int32)[mask] + all_pseudo_gt.to(torch.int32)[mask], 
                                minlength=sem_num ** 2).reshape(sem_num, sem_num)
        # 对角线元素 即为分类正确的
        o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
        tp = torch.diag(histogram) # 真正例
        fp = torch.sum(histogram, 0) - tp # 假正例
        fn = torch.sum(histogram, 1) - tp # 假负例
        IoUs = tp / (tp + fp + fn + 1e-8)*100
        m_IoU = torch.nanmean(IoUs)
        s = '| mIoU {:5.2f} | '.format(m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(IoU)
        self.trainer.logger.info('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("Superpoints/allAcc", o_Acc, current_epoch)
            self.trainer.writer.add_scalar("Superpoints/mIoU", m_IoU, current_epoch)

        # 下面以 聚类后的语义基元 去求对应真值的众数，继而评估精度
        pseudo_class2gt = -torch.ones_like(all_gt)
        for i in range(self.trainer.cluster_cfg.primitive_num):
            mask = all_pseudo==i
            pseudo_class2gt[mask] = torch.mode(all_gt[mask]).values
        mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
        # hungarian matching
        histogram = torch.bincount(sem_num* all_gt.to(torch.int32)[mask] + pseudo_class2gt.to(torch.int32)[mask], 
                                minlength=sem_num ** 2).reshape(sem_num, sem_num)    
        o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
        tp = torch.diag(histogram)
        fp = torch.sum(histogram, 0) - tp
        fn = torch.sum(histogram, 1) - tp
        IoUs = tp / (tp + fp + fn + 1e-8)*100
        m_IoU = torch.nanmean(IoUs)
        s = '| mIoU {:5.2f} | '.format(m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(IoU)
        self.trainer.logger.info('Primitives oAcc {:.2f} IoUs'.format(o_Acc) + s)
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("Primitives/allAcc", o_Acc, current_epoch)
            self.trainer.writer.add_scalar("Primitives/mIoU", m_IoU, current_epoch)
        return classifier.cuda()
    

    @staticmethod
    def get_fixclassifier(in_channel, centroids_num, centroids):
        classifier = torch.nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
        centroids = F.normalize(centroids, dim=1)
        classifier.weight.data = centroids
        for para in classifier.parameters():
            para.requires_grad = False
        return classifier

    @staticmethod
    def compute_hist(normal, bins=10, min=-1, max=1):
        ## normal : [N, 3]
        sample_size = 1000
        if normal.shape[0] > 1000:
            random_indices = torch.randperm(normal.size(0))[:sample_size]
            normal = normal[random_indices, :]
        normal = F.normalize(normal)
        relation = torch.mm(normal, normal.t())
        relation = torch.triu(relation, diagonal=0) # top-half matrix
        hist = torch.histc(relation, bins, min, max)
        # hist = torch.histogram(relation, bins, range=(-1, 1))
        hist /= hist.sum()
        return hist
    
    @staticmethod
    def get_pseudo(args, context, primitive_labels, all_neural_region=None):
        """
        Args:
            context:  (scene_id, gt, raw_region) 构成的 list
            primitive_labels: primitive_labels, 所有train_room 的超点的经过KMeans得到的伪标签，
                        shape 0 = 所有train_room 超点的个数
            all_neural_region: neural_region 构成的list
        Returns:
            all_pseudo: 伪标签
            all_gt: 真值
            all_pseudo_gt: 伪标签 对应的 真值众数
        """
        print('computing pseduo labels...')
        
        all_gt = []
        all_pseudo = []
        all_pseudo_gt = []

        pc_no = 0
        region_num = 0

        for i in range(len(context)):
            scene_id, labels, region, offset = context[i]   # 这个region 为 raw_region 可能带有 -1的值

            # TODO ： 在这里 解构下
            # 核心问题在于 需要 保存 每一个数据的 伪标签

            # index为：pc_no 的超点neural_region
            # 即 每个超点的 neural_region, + region_num
            # 取出pc_no index 对应的 neural_region, 因为neural_region 都是从零开始的，为了统一，
            # 依次累加，这样，保证每个超点 的 编号 唯一，不重复
            sub_primitive_labels = all_neural_region[pc_no]+ region_num 
            valid_mask = region != -1

            labels_tmp = labels[valid_mask]
            pseudo_gt = -torch.ones_like(labels)
            pseudo_gt_tmp = pseudo_gt[valid_mask]

            # 得到 每个room每个点的 伪标签
            # pseudo = -np.ones_like(labels.cpu().numpy()).astype(np.int32)
            pseudo = torch.ones_like(labels, dtype=torch.int32, device=all_neural_region[0].device) * (-1)
            pseudo[valid_mask] = primitive_labels[sub_primitive_labels]

            for p in np.unique(sub_primitive_labels):
                if p != -1:
                    mask = p == sub_primitive_labels
                    # 整体上是一个room 的去就对应真值的众数
                    sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                    pseudo_gt_tmp[mask] = sub_cluster_gt
            pseudo_gt[valid_mask] = pseudo_gt_tmp
            #
            pc_no += 1
            new_region = torch.unique(sub_primitive_labels)
            region_num += len(new_region[new_region != -1])

            ''' f '''
            # pseudo_label_file = scene_id.replace("s3dis", "s3dis_pl") 
            for i in range(len(offset)):
                
                pseudo_label_file_path = scene_id[i].replace(args.dataset.data_root, args.dataset.pl_path) 
                if not os.path.exists(os.path.dirname(pseudo_label_file_path)):
                    os.makedirs(os.path.dirname(pseudo_label_file_path))
                # 这块也对pseudo（对所有train_room的超点聚类成primitive_nums的类） 进行了保存
                if i == 0:
                    torch.save(pseudo[:offset[i]], pseudo_label_file_path) 
                    # print(pseudo[:offset[i]].shape)
                else:
                    torch.save(pseudo[offset[i - 1] : offset[i]], pseudo_label_file_path) 
                    # print(pseudo[offset[i - 1] : offset[i]].shape)

            all_gt.append(labels)
            all_pseudo.append(pseudo)
            all_pseudo_gt.append(pseudo_gt)

        all_gt = torch.concatenate(all_gt)
        all_pseudo = torch.concatenate(all_pseudo)
        all_pseudo_gt = torch.concatenate(all_pseudo_gt)

        return all_pseudo, all_gt, all_pseudo_gt

    @staticmethod
    def get_sp_feature(args, loader, model, current_growsp, mix_mode):
        # TODO: 优化加速
        # self.trainer.cluster_loader  self.trainer.model
        # voxel_size current_growsp w_rgb w_xyz w_norm c_rgb c_shape
        """
        将 dataloader 放到 model 里面 得到 每一点的 特征
            current_growsp: 当前生长的超点的总数

        return:
            point_feats_list:  超点 region_feats  构成的 list, 
                            model 输出 feats 的维度 + rgb 3  + pfg 10
            point_labels_list:  逐点的 label  构成 的list 
            all_neural_region: neural_region 构成的 list  如果 current_growsp 不为None,\
                即当前 region的 超点数量 变化了，会根据计算的 region_feats \
                重新使用KMeans 进行计算 得到新的 region, 即neural_region, \
                若为 None, 实际还是为  raw_region 

            context: (scene_name, gt, raw_region) 构成的 list

        feats :  就是超点的feats
        e.g. 第一个room, 有150个超点, 那么feats 的维度为: (150, )
        """
        print('computing point feats ....')
        point_feats_list = []
        point_labels_list = []
        all_neural_region = []
        model.eval()
        context = []
        with torch.no_grad():
            for _, input_dict in enumerate(loader):
                # scene_name = input_dict["scene_id"]
                # print(f"这是第{_} : {scene_name}个数据")
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                scene_id = input_dict["scene_id"]
                gt = input_dict["segment"].clone()
                raw_region = input_dict["region"].clone()
                offset = input_dict["origin_offset"]

                """mix_up 实现"""
                # if mix_mode:
                #     grid_coord = input_dict["grid_coord"]
                #     feat = input_dict["feat"]
                #     grid_coord_list = []
                #     feat_list = [] # 用于存在单个feat
                #     for i in range(len(offset)):
                #         if i == 0 :
                #             single_feat = feat[0: offset[i]]
                #             single_grid_coord = grid_coord[0: offset[i]]
                #         else:
                #             single_feat = feat[offset[i - 1]: offset[i]]
                #             single_grid_coord = grid_coord[offset[i - 1]: offset[i]]
                #         feat_list.append(single_feat)    
                #         grid_coord_list.append(single_grid_coord)


                #     # 1. 先将 cum_offset 变为 un_cum_offset
                #     un_cum_offset = torch.diff(offset, 
                #                             prepend=torch.tensor([0], device=offset.device, dtype=torch.long))
                #     sample_index = torch.randperm(len(offset))
                #     # 为确保 mix 的对象不能是自己?  感觉应该也可以，这种情况的几率 理论上讲会比较小
                #     # for i in range(len(sample_index))
                #     mix_grid_coord_list = []
                #     mix_feat_list = []
                #     for i in range(len(offset)):
                #         mix_grid_coord_list.append(grid_coord_list[i])
                #         mix_grid_coord_list.append(grid_coord_list[sample_index[i]])
                #         mix_feat_list.append(feat_list[i])
                #         mix_feat_list.append(feat_list[sample_index[i]])

                #     mix_grid_coord = torch.cat(mix_grid_coord_list)
                #     mix_feat = torch.cat(mix_feat_list)

                #     sample_offset = torch.tensor([un_cum_offset[sample_index[i]] for i in range(len(offset))], device=offset.device, dtype=torch.long)
                #     un_cum_add_sample_offset = torch.add(un_cum_offset, sample_offset)
                #     cum_add_sample_offset = torch.cumsum(un_cum_add_sample_offset, dim=0).long() 

                #     mix_input_dict = dict(
                #         grid_coord = mix_grid_coord.to(device=offset.device),
                #         feat = mix_feat.to(device=offset.device),
                #         offset = cum_add_sample_offset.to(device=offset.device),
                #     )

                #     # modified_offset = ClusterClassifier.modify_offset(offset, valid_mask)
                #     # 利用网络提取特征，从 (N, 6) 变到 (N, 128(model输出特征维度))
                #     feats = model(mix_input_dict)   # feats = model(input_dict)  
                #     # print(feats.shape) 
                #     # 需要把 feats 重新变回来

                #     match_feat_list = []
                #     for i in range(len(offset)):
                #         if i == 0:
                #             match_feat = feats[0:offset[i]]
                #         else:
                #             match_feat = feats[cum_add_sample_offset[i - 1]: (un_cum_offset[i] + cum_add_sample_offset[i - 1])]
                #         match_feat_list.append(match_feat)
                #     feats = torch.cat(match_feat_list)

                # else:
                #     feats = model(input_dict)  

                feats = model(input_dict) 
                feats = F.normalize(feats, dim=-1)  #  TODO: 这一步是和 growsp 不一致的地方
                valid_mask = input_dict["region"] != -1
                
                feats = feats[valid_mask]
                
                normals = input_dict["normal"][valid_mask]
                labels = input_dict["segment"][valid_mask]
                region = input_dict["region"][valid_mask].long() # collate_fn 需要重新整理下
                features = input_dict["feat"][valid_mask]

                # 获取单个 room region的 个数
                region_num_list = input_dict["region_num"]
                ##
                pc_rgb = features[:, 3:]
                # features 里面就是 原本没有经过缩放的坐标
                pc_xyz = features[:, 0:3] # * args.voxel_size # 再变回原来的分辨率 
                ##
                region_num = len(torch.unique(region))
                region_corr = torch.zeros(region.size(0), region_num, device="cuda") # (点数， 超点数)
                region_corr.scatter_(1, region.view(-1, 1), 1)
                region_corr = region_corr.cuda()##[N, M]
                per_region_num = region_corr.sum(0, keepdims=True).t()
                # (M, N) * (N, 128)
                region_feats = F.linear(region_corr.t(), feats.t())/per_region_num   
                # 如果current_growsp 不为None, 计算超点 region的 rgb,xyz,norm concat 到region_feats上
                # Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints
                if current_growsp is not None:  ###
                    # 单个 room 才效果比较好，是因为 加入了 超点的 xyz 吗？
                    region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                    region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                    region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                    rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                    region_feats = F.normalize(region_feats, dim=-1)
                    region_feats = torch.cat((region_feats, 
                                              rgb_w*region_rgb, 
                                              xyz_w*region_xyz, 
                                              norm_w*region_norm), dim=-1)
                    # 判断超点 是否要增长， 如果  region 中 超点个数 大于 current_growsp,
                    # 要增长，KMeans 重新聚类 

                    # TODO 这块，改成 按照 单个 room 来聚类
                    # 每个 room 有多少个 超点
                    sp_idx_list = []
                    sp_idx_num = 0
                    for i in range(len(region_num_list) + 1):
                        if i == 0:
                            single_region_feat = region_feats[:region_num_list[i]]
                        elif i != len(region_num_list):
                            single_region_feat = region_feats[region_num_list[i - 1]:region_num_list[i]]
                        else:
                            single_region_feat = region_feats[region_num_list[i - 1]:]

                        assert single_region_feat.size(0) > 0
                        if single_region_feat.size(0) < current_growsp:
                            n_segments = single_region_feat.size(0)
                        else:
                            n_segments = current_growsp
                        # KMeans  返回的只是每一个样本的类别值 从0 开始 到 k 类
                        # TODO: 超点的增长 也得结合 batch size 以及batch中每一个 region的个数
                            
                        # TODO 同时，超点自适应 增长也是在 这儿改
                        single_sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, 
                                                        n_init=5, 
                                                        random_state=0, 
                                                        ).fit_predict(single_region_feat.cpu().numpy())).long().to(region.device)
                        single_sp_idx += sp_idx_num
                        sp_idx_num += n_segments
                        sp_idx_list.append(single_sp_idx)
                    sp_idx = torch.cat(sp_idx_list, 0)
                else:
                    feats = region_feats
                    sp_idx = torch.tensor(range(region_feats.size(0))).to(region.device)
                # 如果聚完类，类别数变动，需要更换新的标签 [0,0,1,1,2], 
                # 如果类别数不变，则neural_region 和 region是等价的
                neural_region = sp_idx[region] 
                pfh = []

                neural_region_num = len(torch.unique(neural_region))
                neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num, device="cuda")
                neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
                neural_region_corr = neural_region_corr.cuda()
                per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
                
                '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
                final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num
                #
                if current_growsp is not None:
                    # (新的超点数, N)*(N, model输出特征维度) = (新的超点数， model 输出特征维度)
                    feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num 
                    feats = F.normalize(feats, dim=-1)

                for p in torch.unique(neural_region):
                    if p!=-1:
                        mask = p==neural_region
                        pfh.append(ClusterClassifier.compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

                pfh = torch.cat(pfh, dim=0)
                feats = F.normalize(feats, dim=-1)
                # #  feats 的维度 + rgb 3  + pfg 10
                feats = torch.cat((feats, args.c_rgb*final_rgb, args.c_shape*pfh), dim=-1)
                feats = F.normalize(feats, dim=-1)

                point_feats_list.append(feats.cpu())
                point_labels_list.append(labels.cpu())

                all_neural_region.append(neural_region.cpu())
                context.append((scene_id, gt.cpu(), raw_region.cpu(), offset.cpu()))

                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch.device("cuda"))
        print('End computing point feats ....')    
        return point_feats_list, point_labels_list, all_neural_region, context

    @staticmethod
    def modify_offset(offset, validmask):
        outlier_counts = []
        for i in range(len(offset)):
            if i == 0:
                cout = torch.sum(validmask[0 : offset[i]] == False).item()
            else:
                cout = torch.sum(validmask[offset[i - 1] : offset[i]] == False).item()
                cout = cout + outlier_counts[-1]
            outlier_counts.append(cout)
        
        modified_offset = offset - torch.tensor(outlier_counts, device=offset.device)
        return modified_offset


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "").split(".")[0]
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            for data_path in self.data_list:
                cache_name = self.get_cache_name(data_path)
                data = torch.load(data_path)
                shared_dict(cache_name, data)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)
