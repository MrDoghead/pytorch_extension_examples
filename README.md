# Using Customized Operations in PyTorch

The easiest way to use <torch/extension.h> in cpp and cuda.

Official tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html

There were two ways of building C++ extensions: using just in time (JIT) or setuptools. The former is lighter.

## JIT

The JIT compilation mechanism provides you with a way of compiling and loading your extensions on the fly by calling a simple function in PyTorch’s API called `torch.utils.cpp_extension.load()`.

The first time you run the code, it will take some time, as the extension is compiling in the background. Since we use the Ninja build system to build your sources, re-compilation is incremental and thus re-loading the extension when you run your Python module a second time is fast and has low overhead if you didn’t change the extension’s source files.

Find the simple cpp example in the folder `example_cpp_jit`

Find the simple mixed cpp/cuda example in the folder `example_cuda_jit`


## Setuptools

Setuptools is a fully-featured, actively-maintained, and stable library designed to facilitate packaging Python projects.

You can write a `setup.py` script to build and install your extension.

Find the simple cpp example in the folder `example_cpp_setuptools`

Find the simple mixed cpp/cuda example in the folder `example_cuda_setuptools`


# Environment

I did tests on an A100 server, and here is my environment:

System: Ubuntu 20.04.6 LTS
Python: 3.10.13
PyTorch: 2.2.1
CUDA Version: 12.2
Driver Version: 535.161.07
gcc/g++/c++: 11.3.0


# Issues

1. compilers: Due to ABI versioning issues, the compiler you use to build your C++ extension must be ABI-compatible with the compiler PyTorch was built with. In practice, this means that you must use GCC version 4.9 and above on Linux. For Ubuntu 16.04 and other more-recent Linux distributions, this should be the default compiler already. On MacOS, you must use clang (which does not have any ABI versioning issues). In the worst case, you can build PyTorch from source with your compiler and then build the extension with that same compiler.

2. ModuleNotFoundError: No module named 'matmul_cpp'

add the path of `build/matmul_cpp.so` to the python path 

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/build
```
