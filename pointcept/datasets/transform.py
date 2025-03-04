import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

import MinkowskiEngine as ME

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        # keys_dict: 一个字典，指定要复制的键值对关系，
        # 键是原始数据字典中的键，值是新数据字典中的键
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Add(object):
    def __init__(self, keys_dict=None):
        # keys_dict: 一个字典，包含要添加的键值对
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict
    
@TRANSFORMS.register_module()
class NormalizeSpColor(object):
    def __call__(self, data_dict):
        if "sp_rgb" in data_dict.keys():
            data_dict["sp_rgb"] = data_dict["sp_rgb"] / 255
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class CentroidShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # data_dict["coord"] = data_dict["coord"].astype(np.float32)
            if self.apply_z:
                # if data_dict["scene_id"] == "data/s3dis/Area_1/office_30.pth":
                #     print(f"scene_id: {data_dict['scene_id']}")
                #     print(f"coord_mean : {data_dict['coord'].mean(axis=0)}")

                data_dict["coord"] -= data_dict["coord"].mean(axis=0).astype(np.float32)
            else:
                coords_center = data_dict["coord"].mean(axis=0).astype(np.float32)
                coords_center[2] = 0.0
                data_dict["coord"] -= coords_center
        return data_dict

@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                # 更新 data_dict["sampled_index"], np.where 没有提供values，只有
                # condition, 只返回满足条件的索引（为True）, 返回值为包含一个array的
                # tuple， 所以后面加了 [0]
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "region" in data_dict.keys():
                data_dict["region"] = data_dict["region"][idx]
            if "pseudo" in data_dict.keys():
                data_dict["pseudo"] = data_dict["pseudo"][idx]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        """
        sigma:随机抖动的标准差，默认为 0.01
        clip:抖动值的截断范围，默认为 0.05
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        # 有点问题 
        # 原本的为: self.mean = np.mean(3)
        self.mean = np.zeros(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            # 1.96 对应标准正态分布中 95%的置信度
            # jitter / 1.96, 并将其结果限制在（-1， 1） 缩放和截断的目的是：
            # 将抖动的范围控制在 标准正态分布中的一个特定置信度内
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            # tr : [-255, 255] * ratio ; 
            # ratio : 0.05, tr: [-12.75, 12.75]
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            # noise 是从 标准正态分布 * std 的分布中进行采样
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)
        coords_max = coords.max(0)

        # Create Gaussian noise tensor of the size given by granularity.
        # 原本的: 
        # noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        # 在每个坐标轴上增加3，确保范围更大，以防止生成的高斯噪声在实际使用中超出边界。
        noise_dim = ((coords_max - coords_min) // granularity).astype(int) + 3
        # 创建一个 3 维 网格，每个网格 x,y,z 共3个噪音元素
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        # 分开进行 x、y、z 方向的卷积可能是为了更灵活地控制每个方向上的平滑程度。
        # 如果将三个方向合并成一个 3x3x3 的卷积核，可能会失去对每个方向单独调整的能力。
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        # 生成 3 维格网坐标点
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        # interp(coords) 对 coords 基于ax 上的 noise 值进行插值
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict

@TRANSFORMS.register_module()
class GridSample_M(object):
    def __init__(
            self,
            grid_size=0.05,
            keys=("coord", "color", "normal", "segment"),
            return_grid_coord=False,
            return_index=False,
            return_inverse=False,
    ):
        self.grid_size = grid_size
        self.keys = keys
        self.return_grid_coord=return_grid_coord
        self.return_index = return_index
        self.return_inverse = return_inverse

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        # coords = data_dict["coord"]
        scaled_coord = data_dict["coord"] / np.array(self.grid_size).astype(float)  
        grid_coord = np.floor(scaled_coord).astype(int)
        # feats = data_dict[""]
        grid_coord_, idx_unique, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(grid_coord), 
                                return_index=self.return_index, 
                                return_inverse=self.return_inverse)
                               # quantization_size=self.grid_size)
        if data_dict["scene_id"] == "data/s3dis/Area_1/office_30.pth":
            print(f"scene_id: {data_dict['scene_id']}")
            print(f"idx_unique.shape : {idx_unique.shape}")
        grid_coord_ = grid_coord_.cpu().numpy()
        idx_unique = idx_unique.cpu().numpy()
        inverse_map = inverse_map.cpu().numpy()
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord_
        if self.return_index:
            data_dict['idx_unique'] = idx_unique
        if self.return_inverse:
            data_dict["inverse"] = inverse_map
            # data_dict["inverse"][idx_sort] = inverse

        for key in self.keys:
            data_dict[key] = data_dict[key][idx_unique]

        return data_dict

@TRANSFORMS.register_module()
class GridSample(object):
    '''这个做法与MinkowskiEngine的 ME.utils.sparse_quantize 量化很相似'''
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_index=False,
        return_inverse=False,
        return_inverse_offset=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.return_inverse_offset = return_inverse_offset

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        # 将self.grid_size 转或者不转成 np.array 都一样,  astype(float) 是 float64
        scaled_coord = data_dict["coord"] / np.array(self.grid_size).astype(float)  
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        # 目的是对不同的坐标生成一个唯一的哈希键，用于对稀疏的坐标进行排序和分组
        # 当然 grid_coord 里面包含具有相同坐标值的坐标
        key = self.hash(grid_coord)
        # 返回按值排序的原始数组的索引
        idx_sort = np.argsort(key)
        # 得到排序的哈希值，值相同的在一块
        key_sort = key[idx_sort]
        # 获取唯一值
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            # idx_select: 决定了在训练模式下，哪些点会被选择用于后续的处理
            # np.cumsum(np.insert(count, 0, 0)[0:-1]) : 是为了获取每个唯一值开始的索引
            # + ... % : 为了在每个累积和的位置加上一个随机的偏移量，确保最终的选择索引是随机的。
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            # idx_unique : 该索引是对于原始数组来讲的
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_index:
                data_dict['idx_unique'] = idx_unique
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_inverse_offset:
                data_dict["inverse_offset"] = torch.tensor([inverse.shape[0]])
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        # 在测试模式下对输入数据进行分割，每个部分的索引通过计算得到，
        # 并且可以选择性地返回其他相关的信息。
        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                # 使用计算得到的索引从原始数据中选择当前部分的索引
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                # 如果需要返回其他信息（如逆向索引、网格坐标、最小坐标、位移等），
                # 则在相应的条件下添加相应的信息到 data_part 字典中。
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        该方法接受一个二维数组 arr 作为输入，然后对坐标进行处理，生成哈希值。
        在数据结构中，哈希值用于快速查找数据
        """
        assert arr.ndim == 2
        # 深拷贝，防止影响原始数组
        arr = arr.copy()
        # 对每列（坐标轴）减去该列的最小值，这样将坐标平移到非负的范围,因为后面要转为 uint64
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        # 目的是为了计算每个坐标轴的哈希值时需要用到每个坐标轴的范围
        # 加 1，是为了避免出现最大值为 0 的情况
        arr_max = arr.max(0).astype(np.uint64) + 1
        # 用于存储计算得到的哈希值
        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing: 按照列
        # 将多维数组中的每一列的值进行混合（组合）生成一个唯一的哈希码
        for j in range(arr.shape[1] - 1):
            # 将当前列的值纳入哈希的计算中
            keys += arr[:, j]
            # 确保每列的值对哈希码的贡献都是唯一的
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        # 返回计算得到的哈希值数组。这个哈希值可以用于唯一地标识输入坐标
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        这个算法的关键点在于通过循环对输入数组的每一列进行异或和乘法操作，
        以确保每个输入值都对最终哈希值产生了影响。这种哈希算法通常用于生成固定长度的哈希码，
        并且在哈希函数的设计中具有一定的性能和分布均匀性优势。
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

@TRANSFORMS.register_module()
class SquareCrop(object):
    def __init__(self, block_size, center=None, return_idx_crop=False):
        self.block_size = block_size
        self.center = center
        self.return_idx_crop = return_idx_crop
    
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        coords = data_dict["coord"]
        bound_min = np.min(coords, axis=0).astype(float)
        bound_max = np.max(coords, axis=0).astype(float)
        bound_size = bound_max - bound_min
        if self.center is None:
            self.center = bound_min + bound_size * 0.5   # ###### 巨坑
        # if data_dict["scene_id"] == "data/s3dis/Area_1/office_30.pth":
        #     print(f"scene_id: {data_dict['scene_id']}")
        #     print(f"bound_min: {bound_min}")
        #     print(f"bound_size: {bound_size}")
        #     print(f"center : {self.center}")
        if isinstance(self.block_size, (int, float)):
            if bound_size.max() < self.block_size:
                return data_dict
            else:
                idx_crop = ((coords[:, 0] >= (-self.block_size + self.center[0])) & \
                            (coords[:, 0] <  ( self.block_size + self.center[0])) & \
                            (coords[:, 1] >= (-self.block_size + self.center[1])) & \
                            (coords[:, 1] <  ( self.block_size + self.center[1])) & \
                            (coords[:, 2] >= (-self.block_size + self.center[2])) & \
                            (coords[:, 2] <  ( self.block_size + self.center[2])))
                # if data_dict["scene_id"] == "data/s3dis/Area_1/office_30.pth":
                #     print(f"scene_id: {data_dict['scene_id']}")
                #     print(f"idx_crop.shape : {idx_crop.shape}")
                #     print(f"idx_crop.sum(): {idx_crop.sum()}")
                if "coord" in data_dict.keys():
                    data_dict["coord"] = data_dict["coord"][idx_crop]
                if "color" in data_dict.keys():
                    data_dict["color"] = data_dict["color"][idx_crop]
                if "normal" in data_dict.keys():
                    data_dict["normal"] = data_dict["normal"][idx_crop]
                if "segment" in data_dict.keys():
                    data_dict["segment"] = data_dict["segment"][idx_crop]
                if "region" in data_dict.keys():
                    data_dict["region"] = data_dict["region"][idx_crop]
                if "instance" in data_dict.keys():
                    data_dict["instance"] = data_dict["instance"][idx_crop]
                if "displacement" in data_dict.keys():
                    data_dict["displacement"] = data_dict["displacement"][idx_crop]
                if "strength" in data_dict.keys():
                    data_dict["strength"] = data_dict["strength"][idx_crop]
                if self.return_idx_crop:
                    data_dict["idx_crop"] = idx_crop
        return data_dict


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random", return_idx_crop=False):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode
        self.return_idx_crop = return_idx_crop

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )
        assert "coord" in data_dict.keys()
        
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                # idx_uni: 用于存储唯一的索引值
                coord_p, idx_uni = np.random.rand(
                    data_dict["coord"].shape[0]
                ) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    # 计算每个点与初始点之间的欧氏距离的平方
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "region" in data_dict.keys(): 
                        data_crop_dict["region"] = data_dict["region"][idx_crop]
                    if "" in data_dict.keys(): # TODO
                        pass
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    # 将当前选择的数据部分的权重设为该部分中每个点与初始点的距离平方
                    # 距离初始点越远，weight 越大
                    data_crop_dict["weight"] = dist2[idx_crop]
                    # 将当前选择的数据部分的索引设为该部分中每个点的原始索引
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    # 更新 coord_p，即每个点的权重估计。加上更新量 delta 会增加被选中的点的权重，
                    # 从而影响下一轮的采样。这个操作确保被选中的点在下一轮采样中具有更低的概率被选中
                    # 计算权重更新量。这里使用了一个归一化的差值，确保权重更新量在 [0, 1] 范围内
                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    ) 
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            # center 模式这种计算方法只针对于 排序好的点云,
            # 目前已改为 其中心
            elif self.mode == "center":
                # center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
                center = np.mean(data_dict["coord"], axis=0)
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "region" in data_dict.keys():
                data_dict["region"] = data_dict["region"][idx_crop]
            if "pseudo" in data_dict.keys():
                data_dict["pseudo"] = data_dict["pseudo"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
            if self.return_idx_crop:
                data_dict["idx_crop"] = idx_crop
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        if "region" in data_dict.keys():
            data_dict["region"] = data_dict["region"][shuffle_index]
        if "pseudo" in data_dict.keys():
            data_dict["pseudo"] = data_dict["pseudo"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        # 将 "segment" 展平，以便生成掩码
        segment = data_dict["segment"].flatten()
        # 生成掩码，排除标签为 0 或 1 的区域
        mask = (segment != 0) * (segment != 1)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][mask]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][mask]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][mask]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][mask]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][mask]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][mask]
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        # 初始化函数，接收参数：
        # view_keys: 用于指定生成对比视图的数据项，如坐标、颜色、法线、原始坐标等。
        # view_trans_cfg: 数据变换的配置，这里使用了一个名为 Compose 的变换集合
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        # 创建两个字典，用于存储第一和第二个视图的数据。
        view1_dict = dict()
        view2_dict = dict()
        # 遍历 view_keys 中指定的数据项，将原始数据复制到两个视图字典中。
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        # 对两个视图字典分别进行数据变换，使用的变换集合由 view_trans_cfg 参数指定。
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class Get_sp_background_mask(object):
    """用于获取经过超点增长之后的 wall_mask"""
    def __init__(
            self,
            l_ratio_thre=0.5,
            h_ratio_thre=0.5,
            w_ratio_thre=0.5,
        ):
        self.l_ratio_thre = l_ratio_thre 
        self.h_ratio_thre = h_ratio_thre 
        self.w_ratio_thre = w_ratio_thre 
    
    def __call__(self, data_dict):
        # 获取room_lwh
        coord_min = np.min(data_dict["coord"], 0)
        coord_max = np.max(data_dict["coord"], 0)
        room_lwh = coord_max - coord_min
        l, w, h = sorted(room_lwh[:2], reverse=True) + [room_lwh[2]]
        room_lwh = np.array([l, w, h])
        z_lwh = data_dict["z_lwh"]
        # 提取 sp_wall_mask
        compare_room_w_mask = z_lwh[:, 0] < room_lwh[1] # shape: (region_num)

        h_ratio = z_lwh[:, 2] / room_lwh[2]
        l_ratio_room_w = z_lwh[:, 0][compare_room_w_mask] / room_lwh[1]
        l_ratio_room_h = z_lwh[:, 0][~compare_room_w_mask] / room_lwh[0]
        l_ratio = np.zeros_like(h_ratio)
        l_ratio[compare_room_w_mask] = l_ratio_room_w
        l_ratio[~compare_room_w_mask] = l_ratio_room_h

        sp_wall_mask = (l_ratio > self.l_ratio_thre) & (h_ratio > self.h_ratio_thre)

        data_dict['sp_wall_mask'] = sp_wall_mask # shape (region_num,)
        # print(f"data_dict['scene_id']: {data_dict['scene_id']}")
        # print(f"data_dict['sp_wall_mask'].shape: {data_dict['sp_wall_mask'].shape}")
        # print(f"data_dict['sp_wall_mask'].sum: {data_dict['sp_wall_mask'].sum()}")

        # 获取 sp_floor_ceil_mask
        l_ratio_2 = z_lwh[:, 0] / room_lwh[0]
        w_ratio = z_lwh[:, 1] / room_lwh[1]

        sp_floor_ceil_mask = (l_ratio_2 > self.l_ratio_thre) & (w_ratio > self.w_ratio_thre)
        data_dict['sp_floor_ceil_mask'] = sp_floor_ceil_mask # shape (region_num,)
        # print(f"data_dict['scene_id']: {data_dict['scene_id']}")
        # print(f"data_dict['sp_floor_ceil_mask'].shape: {data_dict['sp_floor_ceil_mask'].shape}")
        # print(f"data_dict['sp_floor_ceil_mask'].sum: {data_dict['sp_floor_ceil_mask'].sum()}")

        return data_dict


@TRANSFORMS.register_module()
class Get_intersect_sp(object):
    #用于获取两个经过增强以后的超点 交集
    # 应该需要 offset, batch 级别数据
    def __init__(
            self,
            is_sp_attr=False,
        ):
        self.is_sp_attr = is_sp_attr # 判断是否需要处理超点属性
    
    def __call__(self, data_dict):
        view1_sp = data_dict["view1_region"]
        view2_sp = data_dict["view2_region"]
        # view1_unique, view1_cluster, view1_count = torch.unique(
        #     view1_sp, return_inverse=True, return_counts=True)
        """
        计算 view1_sp 和 view2_sp 的超点标签的交集，剔除交集以外的标签，
        并将未剔除的标签变为 consecutive。
        view1_sp 和 view2_sp 的超点标签需要保持对应。
        
        Args:
            view1_sp (np.ndarray): 增强1后的超点标签
            view2_sp (np.ndarray): 增强2后的超点标签
            
        Returns:
            view1_sp_consecutive (np.ndarray): 剔除交集以外的超点标签后的 view1_sp 标签，
            并且标签为 consecutive
            view2_sp_consecutive (np.ndarray): 剔除交集以外的超点标签后的 view2_sp 标签，
            并且标签为 consecutive
        """
        
        # Step 1: 计算 view1_sp 和 view2_sp 的交集
        view1_labels = set(np.unique(view1_sp))  # 获取 view1_sp 中的所有独特标签
        view2_labels = set(np.unique(view2_sp))  # 获取 view2_sp 中的所有独特标签
        common_labels = view1_labels.intersection(view2_labels)  # 计算交集
        common_labels_sorted = sorted(common_labels)  # 默认是升序
        common_labels_id = np.array(list(common_labels_sorted), dtype=int)
        
        # 基于超点交集，对超点属性进行更新
        if self.is_sp_attr:
            # print(f"self.is_sp_attr:{self.is_sp_attr}")
            data_dict["view1_sp_pfh"] = data_dict["view1_sp_pfh"][common_labels_id]
            data_dict["view2_sp_pfh"] = data_dict["view2_sp_pfh"][common_labels_id]

            data_dict["view1_a_lwh"] = data_dict["view1_a_lwh"][common_labels_id]
            data_dict["view2_a_lwh"] = data_dict["view2_a_lwh"][common_labels_id]

            data_dict["view1_o_lwh"] = data_dict["view1_o_lwh"][common_labels_id]
            data_dict["view2_o_lwh"] = data_dict["view2_o_lwh"][common_labels_id]

            data_dict["view1_z_lwh"] = data_dict["view1_z_lwh"][common_labels_id]
            data_dict["view2_z_lwh"] = data_dict["view2_z_lwh"][common_labels_id]

            data_dict["view1_sp_z"] = data_dict["view1_sp_z"][common_labels_id]
            data_dict["view2_sp_z"] = data_dict["view2_sp_z"][common_labels_id]

            data_dict["view1_sp_rgb"] = data_dict["view1_sp_rgb"][common_labels_id]
            data_dict["view2_sp_rgb"] = data_dict["view2_sp_rgb"][common_labels_id]

            if "view1_sp_wall_mask" in data_dict:
                data_dict["view1_sp_wall_mask"] = data_dict["view1_sp_wall_mask"][common_labels_id]
                data_dict["view2_sp_wall_mask"] = data_dict["view2_sp_wall_mask"][common_labels_id]
                data_dict["view1_sp_floor_ceil_mask"] = data_dict["view1_sp_floor_ceil_mask"][common_labels_id]
                data_dict["view2_sp_floor_ceil_mask"] = data_dict["view2_sp_floor_ceil_mask"][common_labels_id]
        
        # Step 2: 只保留交集中的标签
        mask_view1 = np.isin(view1_sp, list(common_labels))  # 对 view1_sp 进行掩码
        mask_view2 = np.isin(view2_sp, list(common_labels))  # 对 view2_sp 进行掩码
        
        view1_sp_filtered = view1_sp[mask_view1]  # 剔除非交集标签
        view2_sp_filtered = view2_sp[mask_view2]  # 剔除非交集标签

        # Step 3: 将标签变为 consecutive（连续编号）
        # 因为交集后的 view1 和 view2 是对应的，直接应用 np.unique 即可
        _, view1_sp_consecutive = np.unique(view1_sp_filtered, return_inverse=True)
        _, view2_sp_consecutive = np.unique(view2_sp_filtered, return_inverse=True) 
            
        # # Step 3: 将标签变为 consecutive（连续编号）
        # unique_labels, consecutive_labels = np.unique(view1_sp_filtered, return_inverse=True)
        # view1_sp_consecutive = consecutive_labels  # view1_sp 重新编号后的标签
        
        # # 根据 view1 的重新编号，确保 view2 的标签也与 view1 对应
        # label_mapping = {old_label: new_label for old_label, new_label in zip(unique_labels, np.unique(consecutive_labels))}
        # view2_sp_consecutive = np.vectorize(label_mapping.get)(view2_sp_filtered)


        # 对 data_dict 中的数据进行筛选
        for k, v in data_dict.items():
            if k.startswith('view1') and data_dict[k].shape[0] == mask_view1.shape[0]:
                data_dict[k] = v[mask_view1]
            if k.startswith('view2') and data_dict[k].shape[0] == mask_view2.shape[0]:
                data_dict[k] = v[mask_view2]

        data_dict['view1_region'] = view1_sp_consecutive
        data_dict['view2_region'] = view2_sp_consecutive
        
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        # # 初始化函数，接收一个配置参数 cfg，用于指定数据变换的顺序和参数。
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        # 遍历配置中的每个变换项，并使用 TRANSFORMS.build 创建相应的变换实例，
        # 添加到 transforms 列表中。
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
