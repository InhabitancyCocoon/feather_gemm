#pragma once

template <unsigned int TILE_SIZE>
__global__ void tiled_matmul(const float *A, const float *B, float *C, const int N,
                             const float alpha, const float beta) {
    int m = N / TILE_SIZE;
    
    // each thread block is responsible for [TILE_SIZE,TILE_SIZE] sub-matrix of C. 
    // Csub is computed by [TILE_SIZE, N] sub-matrix of A and [N, TILE_SIZE] sub-matrix of B.

    __shared__ float A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE * TILE_SIZE];

    

    A += blockIdx.x * TILE_SIZE * N;
    B += blockIdx.y * TILE_SIZE;
    C += blockIdx.x * TILE_SIZE * N + blockIdx.y * TILE_SIZE;

    const unsigned int tx = threadIdx.x / TILE_SIZE;
    const unsigned int ty = threadIdx.x % TILE_SIZE;
    float value = C[tx * N + ty] * beta;

    #pragma unroll
    for (int i = 0; i < m; i++) {
        // load sub-matrix [TILE_SIZE,TILE_SIZE] of A and B into shared mem.
        
        A_tile[tx * TILE_SIZE + ty] = A[tx * N + ty];
        B_tile[tx * TILE_SIZE + ty] = B[tx * N + ty];
        __syncthreads();

        A += TILE_SIZE;
        B += TILE_SIZE * N;
        
        for (int k = 0; k < TILE_SIZE; k++) {
            value += A_tile[tx * TILE_SIZE + k] * B_tile[k * TILE_SIZE + ty] * alpha;
        }
        __syncthreads();
    }
    C[tx * N + ty] = value;
}