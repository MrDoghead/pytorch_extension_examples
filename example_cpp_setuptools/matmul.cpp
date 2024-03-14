#include <torch/extension.h>
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

void matmul_cpp(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int k, int n)
{
    float sum = 0.0f;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int l = 0; l < k; l++)
            {
                sum += fpMatrixA[i * k + l] * fpMatrixB[l * n + j];
            }
            fpMatrixC[i * n + j] = sum;
            sum = 0.0f;
        }
    }
}

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
    matmul_cpp(
              (float*) tensor_A.data_ptr(),
              (float*) tensor_B.data_ptr(),
              (float*) tensor_C.data_ptr(),
              M,
              K,
              N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_matmul", &torch_launch_matmul, "torch_launch_matmul (cpp)");
}
