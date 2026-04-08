from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "gpu_metrics",
        ["gpu_metrics.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include",
            "/usr/lib/pkgconfig/../../include",
        ],
        libraries=["ze_loader"],  # cruical: links the level zero sysman api
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="gpu_metrics",
    version="0.1.0",
    description="Python bindings for Intel GPU Level Zero Metrics",
    ext_modules=ext_modules,
)
