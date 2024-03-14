import torch
import matmul_cuda

M = 32
K = 32
N = 32
A = torch.randn(M, K).cuda()
B = torch.randn(K, N).cuda()
C = torch.zeros((M, N), dtype=torch.float32).cuda()

print("A:", A)
print("B:", B)

matmul_cuda.torch_launch_matmul(A, B, C, M, K, N)

print("C:", C)

torch_C = torch.matmul(A, B)
print("Torch C:", torch_C)

err = C - torch_C
print(f"err mean: {err.mean()}, std: {err.std()}")


