#pragma once

//NOTE: template functions are exceptions from ODR, you should place the definition in header file such that compiler can instantiate it.

template<unsigned int WARP_SIZE>
__global__ void  coalesce_matmul(const float* A, const float* B, float* C, const int N, const float alpha, const float beta) {
    int x = blockIdx.x * WARP_SIZE + threadIdx.x / WARP_SIZE;
    int y = blockIdx.y * WARP_SIZE + threadIdx.x % WARP_SIZE;
    float value = C[x * N + y] * beta;
    if (x < N && y < N) {
        for (int k = 0; k < N; k++) {
            value += A[x * N + k] * B[k * N + y] * alpha;
        }
    }
    C[x * N + y] = value;
}
