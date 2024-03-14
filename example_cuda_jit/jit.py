from torch.utils.cpp_extension import load
mapping_cuda = load(
    name='matmul_cuda', 
    sources=['matmul_cuda.cpp', 'matmul_cuda_kernel.cu'], 
    build_directory="./build",
    verbose=True
    )
help(mapping_cuda)
