'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-04 19:25:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-09-29 17:24:14
FilePath: /Pointcept-main/pointcept/utils/cache.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Data Cache Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import SharedArray

try:
    from multiprocessing.shared_memory import ShareableList
except ImportError:
    import warnings

    warnings.warn("Please update python version >= 3.8 to enable shared_memory")
import numpy as np


def shared_array(name, var=None):
    if var is not None:
        # check exist
        if os.path.exists(f"/dev/shm/{name}"):
            # print(f"{name} 已经存在")
            return SharedArray.attach(f"shm://{name}")
        # create shared_array
        data = SharedArray.create(f"shm://{name}", var.shape, dtype=var.dtype)
        data[...] = var[...]
        data.flags.writeable = False
    else:
        data = SharedArray.attach(f"shm://{name}").copy()
    return data


def shared_dict(name, var=None):
    name = str(name)
    assert "." not in name  # '.' is used as sep flag
    data = {}
    if var is not None:
        assert isinstance(var, dict)
        keys = var.keys()
        # current version only cache np.array
        keys_valid = []
        for key in keys:
            # print(keys) # 
            if isinstance(var[key], np.ndarray):
                keys_valid.append(key)
                # print(key)
        keys = keys_valid

        # ShareableList(sequence=keys, name=name + ".keys")  #####

        try:
            # 尝试附加到已有的共享内存对象
            existing_keys = ShareableList(name=name + ".keys")
            # print(f"Shared memory {name + '.keys'} already exists, attaching to it.")
        except FileNotFoundError:
            # 如果共享内存不存在，则创建共享内存对象
            # print(f"Creating new shared memory for {name + '.keys'}")
            ShareableList(sequence=keys_valid, name=name + ".keys")

        for key in keys:
            if isinstance(var[key], np.ndarray):
                data[key] = shared_array(name=f"{name}.{key}", var=var[key])
    else:
        keys = list(ShareableList(name=name + ".keys"))
        for key in keys:
            data[key] = shared_array(name=f"{name}.{key}")
    return data
