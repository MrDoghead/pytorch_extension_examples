from torch.utils.cpp_extension import load

matmul_cpp = load(
    name='matmul_cpp', 
    sources=['matmul.cpp'], 
    build_directory="./build",
    verbose=True
    )

help(matmul_cpp)
