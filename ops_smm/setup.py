import os
import glob
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from setuptools import setup, find_packages

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")  # 假设源代码都在 src 文件夹中

    # 获取 C++ 和 CUDA 源文件
    source_cpp = glob.glob(os.path.join(extensions_dir, "*.cpp"))  # 如果没有 C++ 文件，可以忽略
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))  # 查找 CUDA 源文件

    # 组合所有源文件
    sources = source_cpp + source_cuda
    extension = CppExtension  # 默认使用 C++ 扩展
    extra_compile_args = {"cxx": []}
    define_macros = []

    # 如果 CUDA 可用，则切换到 CUDA 扩展
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-gencode", "arch=compute_70,code=sm_70",
            "-gencode", "arch=compute_75,code=sm_75",
            "-gencode", "arch=compute_80,code=sm_80",  # 对于 CUDA 11.7 及更高版本
            "-gencode", "arch=compute_86,code=sm_86",
            "-lineinfo",  # 输出详细调试信息
        ]
    else:
        raise NotImplementedError('CUDA is not available or not found.')

    # 确保所有源文件的完整路径
    sources = [os.path.join(extensions_dir, s) for s in sources]

    # 包含的头文件目录
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            name="smm_cuda",  # 模块名称
            sources=sources,  # 指定源文件
            include_dirs=include_dirs,  # 头文件目录
            define_macros=define_macros,  # 定义宏
            extra_compile_args=extra_compile_args,  # 编译选项
        )
    ]
    return ext_modules

setup(
    name="smm_cuda",
    version="1.0",
    author="WeiLong",
    description="Sparse Matrix Multiplication (CUDA)",
    packages=find_packages(),
    ext_modules=get_extensions(),  # 获取扩展模块
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    install_requires=requirements,
)
