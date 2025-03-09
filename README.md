![logo](logo.png)

# Feather GEMM: Toward cuBLAS performance

## Introduction

This is a work log and self-study journey on how to optimize single precision general matrix multiplication on RTX4060 GPU.
I explained some basic cuda concepts and profiler tricks during kernel optimization.
Currently I am not very satisfied with the results, although I did put a lot of effort.

## Build

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Thanks

> [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
>
> [Beating cuBLAS in Single-Precision General Matrix Multiplication](https://salykova.github.io/sgemm-gpu)
>
> [nvidia cuda c++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
>
> [nvidia ncu document](https://docs.nvidia.com/nsight-compute/index.html)
>
> [professional cuda c programming](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
>
> [triton document](https://triton-lang.org/main/index.html)
>
> [efficient gemm](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
>
> [gpu mode](https://www.youtube.com/@GPUMODE)

## TODO

- kernel 8 performs worse than kernel 7, I wonder why.

- kernel 9 double buffering now is right, but I am not satisfied with the performance.

- For now, my kernels only deal with perfect square matrix with no tile quantization.

- Maybe I will integrate PTX code into cuda code with asm().

- warp level matmul, tensor core

- Hopper features: TMA, Asynchrony, check [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) for details.


## TIPS 
- chcp.com 65001 to avoid garbled in windows cmd.

- ncu is not happy with cloud gpu.