from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'inference_tensor_cpp',
        ['inference_tensor.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='inference_tensor_cpp',
    version='0.1.0',
    ext_modules=ext_modules,
    requires=['pybind11'],
)