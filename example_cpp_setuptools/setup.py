from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='matmul_cpp',
    ext_modules=[
        CppExtension('matmul_cpp', ['matmul.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# In this code, CppExtension is a convenience wrapper around setuptools.
# Extension that passes the correct include paths and sets the language of the extension to C++. 
# The equivalent vanilla setuptools code would simply be:
# Extension(
#    name='lltm_cpp',
#    sources=['lltm.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++')
