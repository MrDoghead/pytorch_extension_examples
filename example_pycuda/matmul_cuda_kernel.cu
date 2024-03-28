extern "C" {
__global__ void matmul_kernel(const float* pfMatrixA, const float* pfMatrixB, float* pfMatrixC, int m, int k, int n)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for(int i =0; i < k; i++)
    {
        sum += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
    }
    pfMatrixC[nRow * n + nCol] = sum;
}
}
