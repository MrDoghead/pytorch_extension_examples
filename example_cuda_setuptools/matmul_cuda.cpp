#include <torch/extension.h>
#include "matmul_cuda_kernel.cuh"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void torch_launch_matmul(
                        torch::Tensor &tensor_A,
                        torch::Tensor &tensor_B,
                        torch::Tensor &tensor_C,
                        int M,
                        int K,
                        int N
                        ) 
{
    CHECK_INPUT(tensor_A);
    CHECK_INPUT(tensor_B);
    CHECK_INPUT(tensor_C);
    launch_matmul(
              (float*) tensor_A.data_ptr(),
              (float*) tensor_B.data_ptr(),
              (float*) tensor_C.data_ptr(),
              M,
              K,
              N
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_matmul", &torch_launch_matmul, "torch_launch_matmul (cuda)");
}
