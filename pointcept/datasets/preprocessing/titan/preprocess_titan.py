"""
Preprocessing Script for Titan

Author: Haifeng Zhao (zhaihaifeng@whu.edu.cn)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import torch
import numpy as np
import multiprocessing as mp
from ply import write_ply,read_ply

import open3d as o3d

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def main():
    # patch_path = "titan_32"
    dataset_root = "/home/zhaohaifeng/data/titan/titan_512"
    output_root = "/home/zhaohaifeng/data/titan_pth/titan_512"
    # source_dir = os.path.join(dataset_root, patch_path)
    patch_list = sorted(glob.glob(os.path.join(dataset_root, "*.ply")))
    patch_list = [os.path.splitext(os.path.basename(path))[0] for path in patch_list]
    # Preprocess data.
    print("Processing scenes...")
    # parse_patch_titan(patch_list[0], dataset_root, output_root)
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    pool = ProcessPoolExecutor(max_workers=64)  # peak 110G memory when parsing normal.
    _ = list(
        pool.map(
            parse_patch_titan,
            patch_list,
            repeat(dataset_root),
            repeat(output_root),
        )
    )



def parse_patch_titan(patch_name, dataset_root, output_root):
    '''
        对titan 分块数据进行重新处理：
        保存为字典的形式：
        {
            'coord':
            'color':
            'normal':
            'semantic_gt':
        }
    '''
    patch_path = os.path.join(dataset_root, patch_name) + ".ply"
    data=read_ply(patch_path)
    patch_coords=np.vstack((data['x'], data['y'], data['z'])).T
    patch_colors=np.vstack((data['c1'],data['c2'],data['c3'])).T
    patch_semantic_gts=data['class'].reshape(-1, 1).astype(np.int64)
    save_path = os.path.join(output_root, patch_name) + ".pth"     
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(patch_coords) 
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
    patch_normals = np.array(pcd.normals)

    patch_coords = np.ascontiguousarray(patch_coords)
    patch_colors = np.ascontiguousarray(patch_colors)
    patch_normals = np.ascontiguousarray(patch_normals)
    patch_semantic_gts = np.ascontiguousarray(patch_semantic_gts)
    save_dict = dict(
        coord=patch_coords,
        color=patch_colors,
        normal=patch_normals,
        semantic_gt=patch_semantic_gts,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"{patch_name}已完成！")


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to Stanford3dDataset_v1.2 dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    args = parser.parse_args()



if __name__ == "__main__":
    main()
