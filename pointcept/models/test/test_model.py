import torch
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.engines.defaults import worker_init_fn
import pointcept.utils.comm as comm
from functools import partial
from tensorboardX import SummaryWriter

def build_train_loader(cfg):
    train_data = build_dataset(cfg)

    train_sampler = None

    init_fn = (
        partial(
            worker_init_fn,
            num_workers=1,
            rank=comm.get_rank(),
            seed=1,
        )
        if 1 is not None
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=2,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        collate_fn=partial(point_collate_fn, mix_prob=0),
        pin_memory=True,
        worker_init_fn=init_fn,
        drop_last=True,
        persistent_workers=True,
    )
    return train_loader

writer = SummaryWriter(log_dir="/home/zhaohaifeng/code/model/Pointcept-main/log/")

#  下面就是 给出 model 的配置， data 的 配置 即可

model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="SpUNet-v1m1", in_channels=9, num_classes=20),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)
# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"
train=dict(
    type=dataset_type,
    split="train",
    data_root=data_root,
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(
            type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
        ),
        # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
        dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
        dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
        dict(type="ChromaticJitter", p=0.95, std=0.05),
        # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
        # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        # dict(type="SphereCrop", point_max=100000, mode="random"),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ShufflePoint"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=("coord", "color", "normal"),
        ),
    ],
    test_mode=False,
)



model_builded = build_model(model["backbone"])
model_builded = model_builded.cuda()
train_loader = build_train_loader(train)
data_iterator = enumerate(train_loader)
for iter, input_dict in data_iterator:
    for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
    logits = model_builded(input_dict)
    print(logits.shape)
    # writer.add_graph(model=model_builded, input_to_model=input_dict, verbose=True, use_strict_trace=False)
    break
print("ff")