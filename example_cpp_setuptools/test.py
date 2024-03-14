import torch
import matmul_cpp

M = 32
K = 32
N = 32
A = torch.randn(M, K)
B = torch.randn(K, N)
C = torch.zeros((M, N), dtype=torch.float32)

print("A:", A)
print("B:", B)

matmul_cpp.torch_launch_matmul(A, B, C, M, K, N)

print("C:", C)

torch_C = torch.matmul(A, B)
print("Torch C:", torch_C)

err = C - torch_C
print(f"err mean: {err.mean()}, std: {err.std()}")


