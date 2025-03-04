_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
num_worker = 12
batch_size = 1 # bs: total bs in all gpus
batch_size_val = 1
mix_prob = 0  # 混合增强策略
empty_cache = False
enable_amp = True
# resume=False

# hook
hooks = [
    dict(type="CheckpointLoader"),  # before_train
    dict(type="IterationTimer", warmup_iter=2), # before_train before_epoch before_step after_step
    dict(type="InformationWriter", log_interval = 16), # before_train before_step after_step after_epoch
    dict(type="SemSegEvaluator_U"), # after_epoch  after_train
    dict(type="CheckpointSaver_U", save_freq=None),  # after_epoch
    # dict(type="PreciseEvaluator", test_last=False),
    dict(type="ClusterClassifier") # before_epoch
]

# Trainer
train = dict(type="DefaultTrainer_U")

# model settings
model = dict(
    type="DefaultSegmentorV3",
    backbone=dict(type="MinkUNet14C", in_channels=6, out_channels=0),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 100
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")


# dataset settings
dataset_type = "S3DISDataset_U"
data_root = "data/s3dis"

cluster_cfg = dict(
    start_grow_epoch=38,
    grow_start=300,
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
            dict(type="SquareCrop", block_size=4),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment", "normal", "region"),
                return_grid_coord=True,
                return_inverse=False,
            ),
            # dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5), # np.pi
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.25]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="CentroidShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "segment", "normal", "region", "scene_id"),
                feat_keys=["coord", "color"],
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
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment", "normal", "region"),
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "segment",
                    "origin_segment",
                    "normal", 
                    "region", 
                    "inverse",
                    "scene_id",
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=True,
        ignore_index=12,
    ),
)
