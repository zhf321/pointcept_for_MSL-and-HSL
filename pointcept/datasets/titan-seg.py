# For Titan seg we usually do 13-class segmentation
class_names = ("Impervious Ground", "Grass", "Building", "Tree", "Car", "Power Line", "Bare land")
metainfo = dict(classes=class_names)
dataset_type = "TitanSegDataset"
data_root = "data/Titan/"
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts="points", pts_semantic_mask="semantic_mask")

backend_args = None

num_points = 50000
block_size = 75
# train_area = list(range(1,33))
test_area = [2, 7, 12, 14, 18, 21, 24, 27]
train_area = [x for x in range(1, 33) if x not in test_area]

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,  # 原数据的列数
        use_dim=[0, 1, 2, 3, 4, 5],  # 保留的列
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(
        type="IndoorPatchPointSample",
        num_points=num_points,
        block_size=block_size,
        ignore_index=len(class_names),
        use_normalized_coord=True,
        enlarge_size=0.2,
        min_unique_num=None,
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
        flip_box3d=False,
    ),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(
        type="RandomJitterPoints",
    ),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

tta_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    # dict(type="NormalizePointsColor", color_mean=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(
                    type="RandomFlip3D",
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.0,
                    flip_ratio_bev_vertical=0.0,
                )
            ],
            [dict(type="Pack3DDetInputs", keys=["points"])],
        ],
    ),
]

# train on area 1, 2, 3, 4, 6
# test on area 5
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[f"titan_infos_area_{i}.pkl" for i in train_area],
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=[f"seg_info/area_{i}_resampled_scene_idxs.npy" for i in train_area],
        test_mode=False,
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[f"titan_infos_area_{i}.pkl" for i in test_area],
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=[f"seg_info/area_{i}_resampled_scene_idxs.npy" for i in test_area],
        test_mode=True,
        backend_args=backend_args,
    ),
)
val_dataloader = test_dataloader

val_evaluator = dict(type="SegMetric")
test_evaluator = val_evaluator

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="Visualizer", vis_backends=vis_backends, name="visualizer")

tta_model = dict(type="Seg3DTTAModel")
