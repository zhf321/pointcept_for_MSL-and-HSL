_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 6  # bs: total bs in all gpus  16 
num_worker = 16
mix_prob = 0
empty_cache = False
enable_amp = False
evaluate = False
find_unused_parameters = True
# clip_grad = 1.0

# model settings
model = dict(
    type="SC-v1m1",
    # backbone=dict(
    #     type="SpUNet-v1m1",
    #     in_channels=6,
    #     num_classes=0,
    #     channels=(32, 64, 128, 256, 256, 128, 96, 96),
    #     layers=(2, 3, 4, 6, 2, 2, 2, 2),
    # ),
    backbone=dict(
        type="Res16FPN18",
        in_channels=6,
        out_channels=300,  # primitive_num
        conv1_kernel_size=5,
    ),
    backbone_in_channels=6,
    backbone_out_channels=128, # 96

    view1_mix_prob=0.8,
    view2_mix_prob=0,

    matching_max_pair=4000,  # 2000
    nce_t=0.4,
)

# 在 optimizer 中只设置query 参数需要更新
is_query=True  

# scheduler settings
epoch = 600
optimizer = dict(type="SGD", lr=0.1, momentum=0.8, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
dataset_type = "S3DISDataset_GC_SPA"
data_root = "data/s3dis_S"
hash_code = "ebe5"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"),
        data_root=data_root,
        hash_code=hash_code,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "color", "normal", "origin_coord", "region", # 点级别属性
                           "sp_pfh", "a_lwh", "o_lwh", "z_lwh", 
                           "sp_z", "sp_rgb"), # 超点级别属性
                view_trans_cfg=[
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=1),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=1),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="RandomColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.02,
                        p=0.8,
                    ),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=("origin_coord", "coord", "color", "normal", "region"),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                ],
            ),
            dict(type="Get_intersect_sp"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "view1_origin_coord",
                    "view1_grid_coord",
                    "view1_coord",
                    "view1_color",
                    "view1_normal",
                    "view1_region",
                    "view2_origin_coord",
                    "view2_grid_coord",
                    "view2_coord",
                    "view2_color",
                    "view2_normal",
                    "view2_region",
                ),
                offset_keys_dict=dict(
                    view1_offset="view1_coord", view2_offset="view2_coord"
                ),
                view1_feat_keys=("view1_color", "view1_normal"),
                view2_feat_keys=("view2_color", "view2_normal"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
