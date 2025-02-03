#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include "kernels.cuh"


#define CUDA_CHECK(call) \
{                         \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


void random_initialize(float *data, const int N);

void check_result(const float* data_1, const float* data_2, const int N);

void run_kernel(float* A, float* B, float* C, const int N, const int kernel, const float alpha, const float beta);

void cpu_gemm(const float* A, const float* B, float* C, const int N, const float alpha, const float beta);

void simple_device_query();

void copy_matrix(float* dst, const float* src, const int N);

__global__ void warm_up_kernel(float* C, const int N);
