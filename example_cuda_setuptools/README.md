# Example of using Setuptools

We are going to implement a `matmul` module in mixed cpp/cuda, and install it.

```bash
python setup.py install

# you will see it is install in python site-packages, like
# Installed /root/anaconda3/envs/py310/lib/python3.10/site-packages/matmul_cuda-0.0.0-py3.10-linux-x86_64.egg

python test.py
```