#pragma once


/*
    ncu --section WarpStateStats --metric smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_barrier,
    smsp__pcsamp_warps_issue_stalled_mio_throttle,smsp__pcsamp_warps_issue_stalled_not_selected   matrix_multiply

    From the profile, the main bottleneck of kernel 3 and kernel 4 is: smsp__pcsamp_warps_issue_stalled_mio_throttle.
    It means that there are two much smem instructions.
    Now the optimization is for less smem instructions: 
    one thread computes more element but still load two element in smem(one for tile A, one for tile B).
    This is achieved by changing the size of tile from (32, 32) into (64, 8).
    The tiled matmul is now: 64 x 8(tile A) times 8 x 64(tile B), result in a 64 x 64 tile C.
    One thread now load two element, and compute 8 element rather than one in tile C.
*/



template <unsigned int TILE_ROW, unsigned int TILE_COL, unsigned int elemPerThread>
__global__ void block_tiling_1D_matmul(const float *A, const float *B, float *C,
                                       const int N, const float alpha, const float beta) {
    int m = N / TILE_COL;

    int bx = blockIdx.y, by = blockIdx.x;   // This is better, because the memory access of A is consecutive while sharing submatrix of B.
    // int bx = blockIdx.x, by = blockIdx.y;    // NG, the memory access of B is non-consecutive while sharing submatrix of A.

    // Move A, B to the initial block.
    A += bx * N * TILE_ROW;
    B += by * TILE_ROW;

    C += bx * N * TILE_ROW + by * TILE_ROW;

    int Ax = threadIdx.x / TILE_COL;
    int Ay = threadIdx.x % TILE_COL;
    int Bx = threadIdx.x / TILE_ROW;
    int By = threadIdx.x % TILE_ROW;

    int Cx = threadIdx.x / TILE_ROW;
    int Cy = threadIdx.x % TILE_ROW;

    __shared__ float A_TILE[TILE_ROW][TILE_COL];
    __shared__ float B_TILE[TILE_COL][TILE_ROW];



    float results[elemPerThread]{};


    #pragma unroll
    for (int i = 0; i < m; ++i) {

        A_TILE[Ax][Ay] = A[Ax * N + Ay];
        B_TILE[Bx][By] = B[Bx * N + By];

        __syncthreads();

        A += TILE_COL;
        B += TILE_COL * N;

        // each element compute 8 elements along column dimension in tile C(memory coalesce when write result back into global C.)
        // Thus need one column of B_TILE, 8 rows of A_TILE. It's a (8, 8) matrix times (8, 1) matrix.
        // The result of such matmul is linear combination of columns of the (8, 8) matrix.
        // The outer loop is for iterating elem of column in B_TILE,
        // load it into register so it can be used to multiply with different A_TILE elem.

        for (int j = 0; j < TILE_COL; ++j) {
            float B_tmp = B_TILE[j][Cy];
            for (int k = 0;  k < elemPerThread; ++k) {
                results[k] += A_TILE[Cx * elemPerThread + k][j] * B_tmp;
            }
        }

        // If we do the navie implementation...
        // Surprisingly, performance is the same as the smarter implementation.
        // The compiler unroll the loop since TILE_COL and elemPerThread are all constexpr and does some elimination.

        // for (int k = 0; k < elemPerThread; ++k) {
        //     for (int j = 0; j < TILE_COL; ++j) {
        //         results[k] += A_TILE[Cx * elemPerThread + k][j] * B_TILE[j][Cy];
        //     }
        // }

        __syncthreads();

    }

    
    for (int i = 0; i < elemPerThread; ++i) {
        C[(Cx * elemPerThread + i) * N + Cy] = results[i] * alpha + beta * C[(Cx * elemPerThread + i) * N + Cy];
    }
}