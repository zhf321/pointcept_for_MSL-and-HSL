import torch
from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.engines.defaults import create_ddp_model, worker_init_fn
from functools import partial
import pointcept.utils.comm as comm


def build_train_loader(train_cfg):
        train_data = build_dataset(train_cfg)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = partial(
            worker_init_fn,
            num_workers=1,
            rank=comm.get_rank(),
            seed=2024,
        )
       

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=0),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=False,
            persistent_workers=True,
        )
        return train_loader

def build_cluster_loader(cfg):
    cluster_data = build_dataset(cfg)
    cluster_data.cluster_mode = True # 通过着这种方式将train_loader 变为 cluster_loader

    if comm.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(cluster_data)
    else:
        train_sampler = None

    init_fn = partial(
            worker_init_fn,
            num_workers=1,
            rank=comm.get_rank(),
            seed=2023,
        )


    cluster_loader = torch.utils.data.DataLoader(
        cluster_data,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        sampler=train_sampler,
        collate_fn=partial(point_collate_fn, mix_prob=0),
        pin_memory=True,
        worker_init_fn=init_fn,
        drop_last=False,
        persistent_workers=True,
    )
    return cluster_loader




# model settings
model = dict(
    type="DefaultSegmentorV3",
    backbone=dict(type="MinkUNet34C", in_channels=6, out_channels=13),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)






# dataset settings
dataset_type = "S3DISDataset_U"
data_root = "data/s3dis"

train_cfg=dict(
    type=dataset_type,
    split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
    data_root=data_root,
    sp_path = "data/s3dis_sp",
    pl_path = "exp/s3dis/semseg-minkunet14c-0-base-u-b-more_clu_freq_normal_mix_b4/pseudo_label",
    transform=[
        dict(type="CentroidShift"),
        dict(type="SquareCrop", block_size=4.0),
        # dict(
        #     type="GridSample",
        #     grid_size=0.05,
        #     hash_type="fnv",
        #     mode="train",
        #     keys=("coord", "color", "segment", "normal", "region"),
        #     return_grid_coord=True,
        #     return_inverse=False,
        #     return_inverse_offset=True,
        # ),
        dict(
                type="GridSample_M",
                grid_size=0.05,
                # hash_type="fnv",
                # mode="train",
                keys=("coord", "color", "segment", "normal", "region"),
                return_grid_coord=True,
                # return_inverse_offset=False,
                return_index=True,
                return_inverse=True,
        ), 
    
        # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5), # np.pi
        # dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="x", p=0.5),
        # dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="y", p=0.5),
        # dict(type="RandomScale", scale=[0.8, 1.25]),

        # dict(type="CentroidShift", apply_z=False),
        # dict(type="NormalizeColor"),
        # # dict(type="ShufflePoint"),
        dict(type="ToTensor"),
        # dict(
        #     type="Collect",
        #     keys=("coord", "grid_coord", "color", "segment", "normal", "region", "scene_id", "inverse_offset"),
        #     feat_keys=["coord", "color"],
        # ),
        # dict(
        #         type="Copy",
        #         keys_dict={"offset": "origin_offset"},
        # ),
    ],
    ignore_index=12,
    test_mode=False,
    cluster_mode=False,
    drop_threshold=10,
)

#train_loader = build_train_loader(train_cfg)
#data_0 = train_loader.dataset.prepare_train_data(0)
cluster_loader = build_cluster_loader(train_cfg)
train_loader = build_train_loader(train_cfg)

#data_0 = train_loader.dataset.prepare_train_data(0)
# data_1 = cluster_loader.dataset[18]
print(f"train_loader1: ")
for i, input in enumerate(train_loader):
    pass
print(f"train_loader2: ")
for i, input in enumerate(train_loader):
    pass

print(f"cluster_loader")
for i, input in enumerate(cluster_loader):
    pass



# train_loader = build_train_loader(train_cfg)
# #data_0 = train_loader.dataset.prepare_train_data(0)
# train_0 = train_loader.dataset[0]
# train_0_1 = train_loader.dataset[0]

print("ff")

