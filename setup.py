import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name="droid_backends_alt",
    ext_modules=[
        CUDAExtension(
            "droid_backends_alt",
            include_dirs=[osp.join(ROOT, "thirdparty/lietorch/eigen")],
            sources=[
                "src/droid.cpp",
                "src/droid_kernels.cu",
                "src/correlation_kernels.cu",
                "src/altcorr_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
