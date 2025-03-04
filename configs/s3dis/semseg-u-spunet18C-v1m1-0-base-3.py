_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
num_worker = 6
batch_size = 6 # bs: total bs in all gpus
batch_size_val = 1
batch_size_clu = 4
mix_prob = 1  # 混合增强策略 这块只是 针对 自监督和 有监督，无监督不行，在 cc hook里面
empty_cache = False
enable_amp = False
# resume=False

# hook
hooks = [
    dict(type="CheckpointLoader"),  # before_train
    dict(type="IterationTimer", warmup_iter=2), # before_train before_epoch before_step after_step
    dict(type="InformationWriter", log_interval = 25), # before_train before_step after_step after_epoch
    dict(type="SemSegEvaluator_U", eval_freq=10), # after_epoch  after_train
    dict(type="CheckpointSaver_U", save_freq=10),  # after_epoch   eval_freq 应与 save_freq保持一致
    # dict(type="PreciseEvaluator", test_last=False),
    dict(type="ClusterClassifier", cluster_freq = 10, mix_mode=True) # before_epoch
]

# Trainer
train = dict(type="DefaultTrainer_U")

# model settings
model = dict(
    type="DefaultSegmentorV3",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 192, 192, 128, 128),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 100
max_epoch = 1300
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")


# dataset settings
dataset_type = "S3DISDataset_U"
data_root = "data/s3dis"

cluster_cfg = dict(
    start_grow_epoch=500,
    grow_start=80,
    grow_end=20,
    primitive_num=300,
    feats_dim=128,
    # voxel_size=0.05,
    current_growsp=None,
    w_rgb=5/5,
    w_xyz=1/5,
    w_norm=4/5,
    c_rgb=3,
    c_shape=3,
)

data = dict(
    num_classes=12,
    ignore_index=12,
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
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        sp_path = "data/s3dis_sp",
        pl_path = "pseudo_label",
        transform=[
            dict(type="CentroidShift"),
            dict(type="SquareCrop", block_size=4.0, center=(0, 0, 0)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment", "normal", "region"),
                return_grid_coord=True,
                return_inverse_offset=False,
            ),
            # dict(
            #     type="GridSample_M",
            #     grid_size=0.05,
            #     # hash_type="fnv",
            #     # mode="train",
            #     keys=("coord", "color", "segment", "normal", "region"),
            #     return_grid_coord=True,
            #     # return_inverse_offset=False,
            #     return_index=True,
            #     return_inverse=True,
            # ), 
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1))),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5), # np.pi
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.25]),
            
            dict(type="CentroidShift", apply_z=False),
            dict(type="NormalizeColor"), 
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "segment", "normal", "region", "scene_id"),
                feat_keys=["coord", "color"],
            ),
            dict(
                type="Copy",
                keys_dict={"offset": "origin_offset"},
            ),
        ],
        ignore_index=12,
        test_mode=False,
        cluster_mode=False,
        drop_threshold=10,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CentroidShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment", "normal", "region"),
                return_grid_coord=True,
                return_inverse=True,
                return_inverse_offset=True,
            ),
            dict(type="CentroidShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_segment",
                    "segment",
                    "normal", 
                    "region", 
                    "inverse",
                    "inverse_offset",
                    "scene_id",
                ),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=True,
        ignore_index=12,
    ),
)
