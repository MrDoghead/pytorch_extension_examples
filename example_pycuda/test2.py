"""
pycuda: approach 2
"""
import torch
import pycuda.driver as cuda
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
import numpy as np

class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder,self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_point(self):
        return self.t.data_ptr()

mod = cuda.module_from_file("./matmul_cuda_kernel.cubin")
matmul_cuda = mod.get_function("matmul_kernel")

M = 32
K = 32
N = 32
A = torch.randn(M, K, dtype=torch.float32).cuda()
B = torch.randn(K, N, dtype=torch.float32).cuda()
C = torch.zeros((M, N), dtype=torch.float32).cuda()
print("A:", A)
print("B:", B)

BS = 16
grid_size = ((N+BS-1)//BS, (M+BS-1)//BS, 1)
block_size = (BS, BS, 1)
print(f"grid_size: {grid_size} block_size: {block_size}")
matmul_cuda(
        Holder(A), Holder(B), Holder(C),
        np.int32(M), np.int32(K), np.int32(N),
        grid=grid_size, block=block_size
        )

print("C:", C)

torch_C = torch.matmul(A, B)
print("Torch C:", torch_C)

err = C - torch_C
print(f"err mean: {err.mean()}, std: {err.std()}")


