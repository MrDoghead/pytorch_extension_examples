# Example of the matmul operation using pycuda

Approach1:

```bash
python test.py
```


Approach2:

using cubin file

```bash
nvcc -Xptxas -O3,-v -arch=sm_80 -cubin matmul_cuda_kernel.cu

python test2.py
```
