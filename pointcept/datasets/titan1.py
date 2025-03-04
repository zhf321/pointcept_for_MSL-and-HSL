from os import path as osp
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

# from mmdet3d.registry import DATASETS
from .builder import DATASETS

# from mmdet3d.structures import DepthInstance3DBoxes
# from .det3d_dataset import Det3DDataset
# from .seg3d_dataset import Seg3DDataset
from torch.utils.data import Dataset


class _TitanSegDataset(Seg3DDataset):
    METAINFO = {
        "classes": ("Impervious Ground", "Grass", "Building", "Tree", "Car", "Power Line", "Bare land"),
        "seg_valid_class_ids": (0, 1, 2, 3, 4, 5, 6),
        "seg_all_class_ids": (0, 1, 2, 3, 4, 5, 6),
        "palette": [
            [159, 159, 165],
            [175, 240, 0],
            [254, 0, 0],
            [0, 151, 0],
            [240, 139, 71],
            [10, 77, 252],
            [190, 0, 0],
        ],
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_prefix: dict = dict(pts="points", pts_instance_mask="", pts_semantic_mask=""),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_lidar=True, use_camera=False),
        ignore_index: Optional[int] = None,
        scene_idxs: Optional[Union[np.ndarray, str]] = None,
        test_mode: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs
        )

    def get_scene_idxs(self, scene_idxs: Union[np.ndarray, str, None]) -> np.ndarray:
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        # when testing, we load one whole scene every time
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError("please provide re-sampled scene indexes for training")

        return super().get_scene_idxs(scene_idxs)


@DATASETS.register_module()
class TitanSegDataset(_TitanSegDataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        ann_files: str = "",
        metainfo: Optional[dict] = None,
        data_prefix: dict = dict(pts="points", pts_semantic_mask=""),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_lidar=True, use_camera=False),
        ignore_index: Optional[int] = None,
        scene_idxs: Optional[Union[np.ndarray, str]] = None,
        test_mode: bool = False,
        **kwargs
    ) -> None:
        # make sure that ann_files and scene_idxs have same length
        ann_files = self._check_ann_files(ann_files)
        scene_idxs = self._check_scene_idxs(scene_idxs, len(ann_files))
        super().__init__(
            data_root=data_root,
            ann_file=ann_files[0],
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs[0],
            test_mode=test_mode,
            **kwargs
        )

        datasets = [
            _TitanSegDataset(
                data_root=data_root,
                ann_file=ann_files[i],
                metainfo=metainfo,
                data_prefix=data_prefix,
                pipeline=pipeline,
                modality=modality,
                ignore_index=ignore_index,
                scene_idxs=scene_idxs[i],
                test_mode=test_mode,
                **kwargs
            )
            for i in range(len(ann_files))
        ]

        # data_list and scene_idxs need to be concat
        self.concat_data_list([dst.data_list for dst in datasets])

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def concat_data_list(self, data_lists: List[List[dict]]) -> None:
        """Concat data_list from several datasets to form self.data_list.

        Args:
            data_lists (List[List[dict]]): List of dict containing
                annotation information.
        """
        self.data_list = [data for data_list in data_lists for data in data_list]

    @staticmethod
    def _duplicate_to_list(x: Any, num: int) -> list:
        """Repeat x `num` times to form a list."""
        return [x for _ in range(num)]

    def _check_ann_files(self, ann_file: Union[List[str], Tuple[str], str]) -> List[str]:
        """Make ann_files as list/tuple."""
        # ann_file could be str
        if not isinstance(ann_file, (list, tuple)):
            ann_file = self._duplicate_to_list(ann_file, 1)
        return ann_file

    def _check_scene_idxs(
        self, scene_idx: Union[str, List[Union[list, tuple, np.ndarray]], List[str], None], num: int
    ) -> List[np.ndarray]:
        """Make scene_idxs as list/tuple."""
        if scene_idx is None:
            return self._duplicate_to_list(scene_idx, num)
        # scene_idx could be str, np.ndarray, list or tuple
        if isinstance(scene_idx, str):  # str
            return self._duplicate_to_list(scene_idx, num)
        if isinstance(scene_idx[0], str):  # list of str
            return scene_idx
        if isinstance(scene_idx[0], (list, tuple, np.ndarray)):  # list of idx
            return scene_idx
        # single idx
        return self._duplicate_to_list(scene_idx, num)
