#pragma once

/*
    In kernel 5, each thread compute 8 elements in C, which increase arithmetic
    intensity.
    In this kernel, I am gonna further increase the arithmetic intensity.
    Now tile_A is (128, 8), tile_B is (8, 128), thus tile_C is (128,128).
    The block size is (16 * 16, 1).
    In result, each thread now compute 8 * 8 elements(a sub square matrix) in tile_C while loading
    4 elements into tile_A and 4 elements into tile_B from global memory.

*/



// each thread load LOAD_FACTOR elements into smem for A and B, repectively, 
// and compute a (COMP_FACTOR, COMP_FACTOR) submatrix of C.
template<unsigned int TILE_ROW, unsigned int TILE_COL, unsigned int LOAD_FACTOR, unsigned int COMP_FACTOR>
__global__ void block_tiling_2D_matmul(const float *A, const float *B, float *C,
                                       const int N, const float alpha, const float beta) {
    int m = N / TILE_COL;

    int bx = blockIdx.y, by = blockIdx.x;

    A += bx * N * TILE_ROW;
    B += by * TILE_ROW;

    C += bx * N * TILE_ROW + by * TILE_ROW;

    int Ax = threadIdx.x / TILE_COL;
    int Ay = threadIdx.x % TILE_COL;
    int Bx = threadIdx.x / TILE_ROW;
    int By = threadIdx.x % TILE_ROW;

    int Cx = threadIdx.x / (TILE_ROW / TILE_COL);
    int Cy = threadIdx.x % (TILE_ROW / TILE_COL);



    __shared__ float A_TILE[TILE_ROW][TILE_COL];
    __shared__ float B_TILE[TILE_COL][TILE_ROW];


    float results[COMP_FACTOR][COMP_FACTOR]{};
    float A_tmp[TILE_COL]{};    // use registers to avoid too frequent access of smem.

    #pragma unroll
    for (int i = 0; i < m; ++i) {
        // Load elements from global memory into shared memory, notice coalecse memory access.

        for (int j = 0; j < LOAD_FACTOR; ++j) {
            A_TILE[Ax * LOAD_FACTOR + j][Ay] = A[(Ax * LOAD_FACTOR + j) * N + Ay];
            B_TILE[Bx * LOAD_FACTOR + j][By] = B[(Bx * LOAD_FACTOR + j) * N + By];
        }


         __syncthreads();

        A += TILE_COL;
        B += TILE_COL * N;

        // Compute 8 x 8 submatrix.
        for (int j = 0; j < COMP_FACTOR; ++j) {

            // fill column j.
            for (int k = 0; k < COMP_FACTOR; ++k) {
                A_tmp[k] = A_TILE[Cx * COMP_FACTOR + k][j];
            }

            // iterate over row j. 
            // Think matmul as linear combination of columns.
            for (int k = 0; k < COMP_FACTOR; ++k) {
                float b_tmp = B_TILE[j][Cy * COMP_FACTOR + k];
                for (int t = 0; t < COMP_FACTOR; ++t) {
                    results[t][k] += b_tmp * A_tmp[t];
                }
            }

        }

        __syncthreads();
    }

    for (int i = 0; i < COMP_FACTOR; ++i) {
        for (int j = 0; j < COMP_FACTOR; ++j) {
            C[(Cx * COMP_FACTOR + i) * N + Cy * COMP_FACTOR + j] = results[i][j] * alpha + beta * C[(Cx * COMP_FACTOR + i) * N + Cy * COMP_FACTOR + j];
        }
    }


}