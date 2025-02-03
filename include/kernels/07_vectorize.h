#pragma once



/*
    From the ncu profiler, 
    ncu --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section SourceCounters  -o kernel_07 main.exe 7
    I can tell that the two main bottleneck now is 
        uncoalesced global memory store
        bank conflicts of shared memory
    I will try to analyze why and try to optimize in kernel 8.
*/



// each thread load LOAD_FACTOR elements into smem for A and B, repectively, 
// and compute a (COMP_FACTOR, COMP_FACTOR) submatrix of C.
// reinterpret_cast is not happy with const qualifiers.
template<unsigned int TILE_ROW, unsigned int TILE_COL, unsigned int LOAD_FACTOR, unsigned int COMP_FACTOR>
__global__ void vectorize(float *A, float *B, float *C,
                                       const int N, const float alpha, const float beta) {
    

    int m = N / TILE_COL;

    int bx = blockIdx.y, by = blockIdx.x;

    A += bx * N * TILE_ROW;
    B += by * TILE_ROW;

    C += bx * N * TILE_ROW + by * TILE_ROW;

    int Ax = threadIdx.x / (TILE_COL / LOAD_FACTOR);
    int Ay = threadIdx.x % (TILE_COL / LOAD_FACTOR);
    int Bx = threadIdx.x / (TILE_ROW / LOAD_FACTOR);
    int By = threadIdx.x % (TILE_ROW / LOAD_FACTOR);

    int Cx = threadIdx.x / (TILE_ROW / TILE_COL);
    int Cy = threadIdx.x % (TILE_ROW / TILE_COL);


    // transpose A_TILE
    __shared__ float A_TILE[TILE_COL][TILE_ROW];
    __shared__ float B_TILE[TILE_COL][TILE_ROW];


    float results[COMP_FACTOR][COMP_FACTOR]{};
    float A_tmp[TILE_COL]{};    // use registers to avoid too frequent access of smem.

    #pragma unroll
    for (int i = 0; i < m; ++i) {
        // Load elements from global memory into shared memory, notice coalecse memory access.

        // vectorize memory access, use reinterpret_cast. I know what I am doing here.
        float4 A_four = reinterpret_cast<float4*>(&A[Ax * N + Ay * LOAD_FACTOR])[0];
        A_TILE[Ay * LOAD_FACTOR + 0][Ax] = A_four.x;
        A_TILE[Ay * LOAD_FACTOR + 1][Ax] = A_four.y;
        A_TILE[Ay * LOAD_FACTOR + 2][Ax] = A_four.z;
        A_TILE[Ay * LOAD_FACTOR + 3][Ax] = A_four.w;
        reinterpret_cast<float4*>(&B_TILE[Bx][By * LOAD_FACTOR])[0] = reinterpret_cast<float4*>(&B[Bx * N + By * LOAD_FACTOR])[0];



         __syncthreads();

        A += TILE_COL;
        B += TILE_COL * N;

        // Compute 8 x 8 submatrix.
        for (int j = 0; j < COMP_FACTOR; ++j) {

            // fill column j.
            for (int k = 0; k < COMP_FACTOR; ++k) {
                A_tmp[k] = A_TILE[j][Cx * COMP_FACTOR + k];
            }

            // iterate over row j. 
            // Think matmul as linear combination of columns of A.
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