"""Install Intrinsic SAID."""
import os
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.5;3.7;6.1;7.0;7.5;8.6+PTX"


def setup_package():
    long_description = "nicl"
    setuptools.setup(
        ext_modules=[
            CUDAExtension(
                "src.models.fwh_cuda",
                sources=[
                    "src/models/fwh_cuda/fwh_cpp.cpp",
                    "src/models/fwh_cuda/fwh_cu.cu",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    setup_package()
