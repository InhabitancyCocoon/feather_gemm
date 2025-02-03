#pragma once

/*
    In kernel 7, each thread compute a (8,8) submatrix of C.
    Because of the stride between threads, the global memory access pattern is a total shit.
    The main bottleneck of kernel 7 is store global memory and bank conflict of shared memory.
    To optimize global store memory access pattern, I need to change how threads compute 64 elements of C.
    I think warp level gemm is now what I needed to achieve coalesced global store.
    Warp level gemm may also solve the bank conflict problem.
    I have a sketch in my head.
    Please check https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md.

    Bank conflicts only happen within a warp.

    Still the block size is 16 * 16, each thread compute 8 elements in C and load 4 elements from A and B into
    shared memory respectively.
    Shared memory has 32 memory banks and the stride is 32 bits(4 btyes or a float).
    The block now has 8 warps.
    Each warp load 32 elements and compute 8 * 32 = 256 elements in C.


    Think of global memory store of C first.
    Each block compute 128 * 128 sub-square matrix of C.
    Now each warp computes a 2 * 128 sub-matrix of C and repeat 16 times such that each warp computes a 32 * 128 submatrix of C.
    The warp computes 32 consecutive elements of C first and moves to the next 32 consecutive elements.
    The compuatation of 1 * 128 submatrix of C is derived by 1 * 8 submatrix of A times 8 * 128 submatrix of B.
    The computation of 1 * 32 submatrix of C is derived by 1 * 8 submatirx of A times 8 * 32 submatrix of B.
    Thus each thread will need 8 registers to store 8 elements of A through the computation of the 1 * 128 submatrix.
    During each computation of the 1 * 32 submatrix, each thread now need to load 8 elements(in a column) from the shared memory of B.


    Second, consider the memeory layout of shared memory. Think B_TILE first.
    The B_TILE is of size 8 * 128. Each float takes a bank.
    So the memory layout of the first row is:
    B0-B31  B0-B31  B0-B31  B0-B31
    The pattern is repeated 7 times more for the below 7 rows.
    During the compuation of 1 * 32 matrix, each thread need to load its respective column of B_TILE.
    This does not cause bank conflict because each thread access its repective bank to load the float.


    Now consider how things are going for A_TILE.
    The main problem is wheather or not to transpose A_TILE.
    8 * 128 or 128 * 8, which one is better?
    Consider the global load from A first.
    The A_TILE in the A is a 128 * 8 submatrix.
    To achieve coalesced memory access, the best memory load pattern is:
    The first 8 elements load 8 elements in a row in A.
    The second 8 elements load the next row.
    As described above, each warp loads a 4 * 8 submatrix of A.
    This pattern is repeated 4 times such that A_TILE can be filled.

    In this pattern, each warp needs to issue a 32 bytes(8 float coalesced) memory instruction four times for each load.
    And that is best I can do.


    The only thing left to consider is the wheather or not to transpose A_TILE.
    If not, which is a 128 * 8 A_TILE, the memory layout is:
    B0 - B7
    B8 - B15
    B16 - B23
    B24 - B31
    If transposed, the memory layout is:
    B0-B31  B0-B31  B0-B31  B0-B31 (which is the same as B_TILE).

    i) Comparison of loading elements into shared memory.
    The two pattern are equivalent considering bank conflict.
    Both do not cause bank conflict.
    ii) Comparison of loading elements from shared memory.
    When computing of 1 * 128 sub-matrix of C_TILE, each thread needs to load 8 elements from A_TILE into registers.
    But each float is a word in memory bank.
    The 32 threads load the same word from shared memory and repeat 8 times.
    This does not generate a bank conflict.

    After the comparison, I can tell that the two pattern is the same. So I decide to not to transpose A_TILE for simplicity.


    How can I time the kernel (as accurately as possible) ?
*/

// OPTIMIZE: How can I autotune the config ? My code is hard code.



template<unsigned int TILE_ROW, unsigned int TILE_COL>
__global__ void warp_tile(float *A, float *B, float *C, const int N,
                          const float alpha, const float beta) {
    int m = N / TILE_COL;

    int bx = blockIdx.y, by = blockIdx.x;

    A += bx * N * TILE_ROW;
    B += by * TILE_ROW;

    C += bx * N * TILE_ROW + by * TILE_ROW;

    int warpIdx = threadIdx.x / 32;
    int Ax = (threadIdx.x % 32) / TILE_COL;
    int Ay = (threadIdx.x % 32) % TILE_COL;

    int Bx = 0;
    int By = threadIdx.x % 32;



    __shared__ float A_TILE[TILE_ROW][TILE_COL];
    __shared__ float B_TILE[TILE_COL][TILE_ROW];

    float A_tmp[TILE_COL]{};

    float results[64]{};


    #pragma unroll
    for (int i = 0; i < m; ++i) {
        
        for (int j = 0; j < 4; ++j) {
            A_TILE[Ax + warpIdx * 16 + j * 4][Ay] = A[(Ax + warpIdx * 16 + j * 4) * N + Ay];
            B_TILE[warpIdx][By + j * 32] = B[warpIdx * N + By + j * 32];
        }

        __syncthreads();

        A += 8;
        B += 8 * N;

        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 8; ++k) {
                A_tmp[k] = A_TILE[warpIdx * 16 + j][k];
            }

            for (int k = 0; k < 4; ++k) {

                for (int t = 0; t < 8; ++t) {
                    results[j * 4 + k] += A_tmp[t] * B_TILE[t][k * 32 + By];
                }
            }

        }

        __syncthreads();



    }


    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[(warpIdx * 16 + i) * N + j * 32 + By] = results[i * 4 + j] * alpha +
                                                      C[(warpIdx * 16 + i) * N + j * 32 + By] * beta;
        }
    }

}