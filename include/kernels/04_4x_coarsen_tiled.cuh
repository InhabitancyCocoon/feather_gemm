#pragma once

template <unsigned int TILE_SIZE, unsigned int COARSEN_FACTOR>
__global__ void tiled_coarsened_matul_4x(const float *A, const float *B, float *C, const int N,
                                      const float alpha, const float beta) {
    int m = N / TILE_SIZE / COARSEN_FACTOR;
    int tx = threadIdx.x / TILE_SIZE, ty = threadIdx.x % TILE_SIZE;

    A += blockIdx.x * COARSEN_FACTOR * TILE_SIZE * N;
    B += blockIdx.y * COARSEN_FACTOR * TILE_SIZE;
    C += blockIdx.x * COARSEN_FACTOR * TILE_SIZE * N + blockIdx.y * COARSEN_FACTOR * TILE_SIZE;

    __shared__ float A_tile[TILE_SIZE * COARSEN_FACTOR][TILE_SIZE * COARSEN_FACTOR];
    __shared__ float B_tile[TILE_SIZE * COARSEN_FACTOR][TILE_SIZE * COARSEN_FACTOR];


    float value_00 = 0;
    float value_01 = 0;
    float value_10 = 0;
    float value_11 = 0;


    for (int i = 0; i < m; i++) {
        A_tile[tx][ty] = A[tx * N  + ty];
        A_tile[tx][ty + TILE_SIZE] = A[tx * N  + ty + TILE_SIZE];
        A_tile[tx + TILE_SIZE][ty] = A[(tx + TILE_SIZE) * N + ty];
        A_tile[tx + TILE_SIZE][ty + TILE_SIZE] = A[(tx + TILE_SIZE) * N + ty + TILE_SIZE];

        B_tile[tx][ty] = B[tx * N  + ty];
        B_tile[tx][ty + TILE_SIZE] = B[tx * N  + ty + TILE_SIZE];
        B_tile[tx + TILE_SIZE][ty] = B[(tx + TILE_SIZE) * N + ty];
        B_tile[tx + TILE_SIZE][ty + TILE_SIZE] = B[(tx + TILE_SIZE) * N + ty + TILE_SIZE];

        A += TILE_SIZE * COARSEN_FACTOR;
        B += TILE_SIZE * COARSEN_FACTOR * N;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE * COARSEN_FACTOR; ++j) {
            value_00 += A_tile[tx][j] * B_tile[j][ty];
            value_01 += A_tile[tx][j] * B_tile[j][ty + TILE_SIZE];
            value_10 += A_tile[tx + TILE_SIZE][j] * B_tile[j][ty];
            value_11 += A_tile[tx + TILE_SIZE][j] * B_tile[j][ty + TILE_SIZE];   
        }
        __syncthreads();
    }

    C[tx * N + ty] = value_00 * alpha + beta * C[tx * N + ty];
    C[tx * N + ty + TILE_SIZE] = value_01 * alpha + beta * C[tx * N + ty + TILE_SIZE];
    C[(tx + TILE_SIZE) * N + ty] = value_10 * alpha + beta * C[(tx + TILE_SIZE) * N + ty];
    C[(tx + TILE_SIZE) * N + ty + TILE_SIZE] = value_11 * alpha + beta * C[(tx + TILE_SIZE) * N + ty + TILE_SIZE];
}