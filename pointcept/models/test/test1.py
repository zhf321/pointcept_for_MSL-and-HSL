import einops
import numpy as np
from einops import rearrange, reduce, repeat, einsum, parse_shape, asnumpy
import inspect
import torch.nn.functional 
aaa = 666

from . import test_inspect

from .test_inspect import example_function

def register_module(self):
    def _register_module(cls_obj):
        self._module_dict[cls_obj.__name__] = cls_obj
        return cls_obj
    return _register_module


print(dir()) # 示例1：查看当前作用域中的所有名称

print(dir(42)) # 示例2：查看整数对象的属性和方法

print(dir("Hello")) # 示例3：查看字符串对象的属性和方法

print(dir([1, 2, 3])) # 示例4：查看列表对象的属性和方法

# 示例5：查看自定义类的属性和方法
class MyClass:
    def __init__(self):
        self.my_attribute = "Hello"

    def my_method(self):
        print("Method called")

obj = MyClass()
print(dir(obj))



def v2():
    print(inspect.stack())

def v1():
    v2()

def main():
    v1()

if __name__ == "__main__":
    main()


def infer_scope():
    """Infer the scope of registry.

    The name of the package where registry is defined will be returned.

    Example:
        # in mmdet/models/backbone/resnet.py
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        The scope of ``ResNet`` will be ``mmdet``.


    Returns:
        scope (str): The inferred scope name.
    """
    # inspect.stack() trace where this function is called, the index-2
    # indicates the frame where `infer_scope()` is called
    filename = inspect.getmodule(inspect.stack()[2][0]).__name__
    split_filename = filename.split(".")
    return split_filename[0]

# name1 = infer_scope()

model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="MinkUNet34C", in_channels=9, out_channels=20),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

print(model)
"""
    einops.rearrange is a reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    Examples for rearrange operation:

    ```python
    # suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> images = [np.random.randn(30, 40, 3) for _ in range(32)]

    # stack along first (batch) axis, output is a single array
    >>> rearrange(images, 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    # concatenate images along height (vertical axis), 960 = 32 * 30
    >>> rearrange(images, 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    # concatenated images along horizontal axis, 1280 = 32 * 40
    >>> rearrange(images, 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    # reordered axes to "b c h w" format for deep learning
    >>> rearrange(images, 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    # flattened each image into a vector, 3600 = 30 * 40 * 3
    >>> rearrange(images, 'b h w c -> b (c h w)').shape
    (32, 3600)

    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
    >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    (128, 15, 20, 3)

    # space-to-depth operation
    >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    (32, 15, 20, 12)

"""

images = [np.random.randn(30, 40, 3) for _ in range(32)]

rearrange(images, 'b h w c -> b h w c').shape
rearrange(images, 'b h w c -> (b h) w c').shape