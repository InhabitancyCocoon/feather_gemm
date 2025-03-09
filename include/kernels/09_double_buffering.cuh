#pragma once



/*
    This is the last kernel I optimiz for sgemm.
    I profile kernel 8, sadly it is rather slower than kernel 7.
    I solved the bank conflict completely and optimize the global store in to C but still kernel 8 's performance
    is bad.
    The ncu said:
    On average, each warp of this kernel spends 2.1 cycles being stalled waiting for the micro scheduler 
    to select the warp to issue. 
    Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another 
    warp was selected. 
    A high number of not selected warps typically means you have sufficient warps to cover warp latencies 
    and you may consider reducing the number of active warps to possibly increase cache coherence and data locality. 
    This stall type represents about 31.2% of the total average of 6.6 cycles between issuing two instructions.

    I compare the warp state statistics, the metric:

    Kernel          Warp Cycles Per Issued Instruction [cycle]
    cublas          5.16
    vectorize       5.95
    warp_tile       6.60

    The metric `Warp Cycles Per Issued Instruction` reflects latency for micro scheduler to select warps.

    As illustrated by cutlass doc:

    The blocked structure demands a large storage allocation within the registers of each CUDA thread. 
    The accumulator elements typically occupy at least half a thread's total register budget. 
    Consequently, occupancy -- the number of concurrent threads, warps, and threadblocks -- is relatively low compared to other classes of GPU workloads. 
    This limits the GPU's ability to hide memory latency and other stalls by context switching to other concurrent threads within an SM.

    To mitigate the effects of memory latency, CUTLASS uses software pipelining to overlap memory accesses with other computation within a thread. CUTLASS accomplishes this by double buffering at the following scopes.

    Threadblock-scoped shared memory tiles: two tiles are allocated in shared memory. 
    One is used to load data for the current matrix operation, while the other tile is used to buffer data loaded from global memory for the next mainloop iteration.

    Warp-scoped matrix fragments: 
    two fragments are allocated within registers. One fragment is passed to CUDA and TensorCores during the current matrix computation, while the other is used to receive shared memory fetch returns for the next warp-level matrix operation.


    Two level double buffering: shared memory double buffering and register double buffering are used to hide memory latency,
    and further reduce warp stall time.

    Double buffering takes 1 times more smem usage and more registers.

    The smem is sufficient but I am worried about register spilling since each thread can use at most 255 registers.

    This is the last try and the last trick, I hope it works.


    Pseudo code looks like this:

    __shared__ float A_TILE[2][size];
    __shared__ float B_TILE[2][size];

    load_from_global(A_TILE[0]);
    load_from_global(B_TILE[0]);

    int prefetch_idx = 1, compute_idx = 0;


    for (int step = 0; step < K / TILE_K - 1; ++step) {
        load_from_global(A_TILE[prefetch_idx]);
        load_from_global(B_TILE[prefethc_idx]);
        compute_tile_matmul(A_TILE[compute_idx], B_TILE[compute_idx]);
        prefetch_idx = 1 - prefetch_idx;
        compute_idx = 1 - compute_idx;
    }

    compute_tile_matmul(A_TILE[compute_idx], B_TILE[compute_idx]);
*/

//Fixme: this kernel now is right, but the performance is bad.

// template<unsigned int TILE_ROW, unsigned int TILE_COL>
// __global__ void double_buffering(float *A, float *B, float *C, const int N,
//                           const float alpha, const float beta) {
//     int m = N / TILE_COL;

//     int bx = blockIdx.y, by = blockIdx.x;

//     A += bx * N * TILE_ROW;
//     B += by * TILE_ROW;

//     C += bx * N * TILE_ROW + by * TILE_ROW;

//     int warpIdx = threadIdx.x / 32;
//     int Ax = (threadIdx.x % 32) / TILE_COL;
//     int Ay = (threadIdx.x % 32) % TILE_COL;

//     int Bx = 0;
//     int By = threadIdx.x % 32;


//     // use one time more smem for prefetch.
//     __shared__ float A_TILE[2 * TILE_ROW * TILE_COL];
//     __shared__ float B_TILE[2 * TILE_COL * TILE_ROW];


//     // use one time more register for prefetch.
//     // It seems that register spilling happens if I used one time more registers..
//     float A_tmp[2 * TILE_COL];


//     float results[64]{};



//     int cur_tile_idx = 0;
//     int cur_a_tmp_idx = 0;


//     // fetch the first tile.
//     #pragma unroll
//     for (int j = 0; j < 4; ++j) {
//         A_TILE[cur_tile_idx * 1024 + (Ax + warpIdx * 16 + j * 4) * TILE_COL + Ay] = A[(Ax + warpIdx * 16 + j * 4) * N + Ay];
//         B_TILE[cur_tile_idx * 1024 + warpIdx * TILE_ROW + By + j * 32] = B[warpIdx * N + By + j * 32];
//     }

    

//     __syncthreads();

//     A += 8;
//     B += 8 * N;

    



//     #pragma unroll
//     for (int i = 1; i < m; ++i) {

//         int next_tile_idx = 1 - cur_tile_idx;

//         // prefetch the next A_TILE and B_TILE.
//         #pragma unroll
//         for (int j = 0; j < 4; ++j) {
//             A_TILE[next_tile_idx * 1024 + (Ax + warpIdx * 16 + j * 4) * TILE_COL + Ay] = A[(Ax + warpIdx * 16 + j * 4) * N + Ay];
//             B_TILE[next_tile_idx * 1024 + warpIdx * TILE_ROW + By + j * 32] = B[warpIdx * N + By + j * 32];
//         }



//         A += 8;
//         B += 8 * N;
        
//         // fetch the first A_tmp
//         #pragma unroll
//         for (int k = 0; k < 8; ++k) {
//             A_tmp[cur_a_tmp_idx * 8 + k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16) * TILE_COL + k];
//         }


//         #pragma unroll
//         for (int j = 0; j < 15; ++j) {

//             int next_a_tmp_idx = 1 - cur_a_tmp_idx;

//             // prefetch the next A_tmp.
//             #pragma unroll
//             for (int k = 0; k < 8; ++k) {
//                 A_tmp[next_a_tmp_idx * 8 + k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16 + j + 1) * TILE_COL + k];
//             }

//             // compute

//             #pragma unroll
//             for (int k = 0; k < 4; ++k) {

//                 #pragma unroll
//                 for (int t = 0; t < 8; ++t) {
//                     results[j * 4 + k] += A_tmp[cur_a_tmp_idx * 8 + t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
//                 }
//             }

            
//             cur_a_tmp_idx = next_a_tmp_idx;

//         }

//         // compute the last A_tmp
//         #pragma unroll
//         for (int k = 0; k < 4; ++k) {

//             #pragma unroll
//             for (int t = 0; t < 8; ++t) {
//                 results[15 * 4 + k] += A_tmp[cur_a_tmp_idx * 8 + t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
//             }
//         }

        

//         cur_tile_idx = next_tile_idx;

//         __syncthreads();

//     }


//     // deal with the last A_TILE and B_TILE.


//     // fetch the first A_tmp
//     #pragma unroll
//     for (int k = 0; k < 8; ++k) {
//         A_tmp[cur_a_tmp_idx * 8 + k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16) * TILE_COL + k];
//     }


//     #pragma unroll
//     for (int j = 0; j < 15; ++j) {

//         // compute
//         #pragma unroll
//         for (int k = 0; k < 4; ++k) {

//             #pragma unroll
//             for (int t = 0; t < 8; ++t) {
//                 results[j * 4 + k] += A_tmp[cur_a_tmp_idx * 8 + t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
//             }
//         }

//         // prefetch the next A_tmp.

//         int next_a_tmp_idx = 1 - cur_a_tmp_idx;

//         #pragma unroll
//         for (int k = 0; k < 8; ++k) {
//             A_tmp[next_a_tmp_idx * 8 + k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16 + j + 1) * TILE_COL + k];
//         }

//         cur_a_tmp_idx = next_a_tmp_idx;

//     }

//     // compute using the last A_tmp
//     #pragma unroll
//     for (int k = 0; k < 4; ++k) {

//         #pragma unroll
//         for (int t = 0; t < 8; ++t) {
//             results[15 * 4 + k] += A_tmp[cur_a_tmp_idx * 8 + t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
//         }
//     }


//     // write result back.


//     #pragma unroll
//     for (int i = 0; i < 16; ++i) {

//         #pragma unroll
//         for (int j = 0; j < 4; ++j) {
//             C[(warpIdx * 16 + i) * N + j * 32 + By] = results[i * 4 + j] * alpha +
//                                                       C[(warpIdx * 16 + i) * N + j * 32 + By] * beta;
//         }
//     }

// }



template<unsigned int TILE_ROW, unsigned int TILE_COL>
__global__ void double_buffering(float *A, float *B, float *C, const int N,
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


    // use one time more smem for prefetch.
    __shared__ float A_TILE[2 * TILE_ROW * TILE_COL];
    __shared__ float B_TILE[2 * TILE_COL * TILE_ROW];


    // use one time more register for prefetch.
    // It seems that register spilling happens if I used one time more registers..
    float A_tmp[TILE_COL];


    float results[64]{};



    int cur_tile_idx = 0;


    // fetch the first tile.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        A_TILE[cur_tile_idx * 1024 + (Ax + warpIdx * 16 + j * 4) * TILE_COL + Ay] = A[(Ax + warpIdx * 16 + j * 4) * N + Ay];
        B_TILE[cur_tile_idx * 1024 + warpIdx * TILE_ROW + By + j * 32] = B[warpIdx * N + By + j * 32];
    }

    

    __syncthreads();

    A += 8;
    B += 8 * N;

    



    #pragma unroll
    for (int i = 1; i < m; ++i) {

        int next_tile_idx = 1 - cur_tile_idx;

        // prefetch the next A_TILE and B_TILE.
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            A_TILE[next_tile_idx * 1024 + (Ax + warpIdx * 16 + j * 4) * TILE_COL + Ay] = A[(Ax + warpIdx * 16 + j * 4) * N + Ay];
            B_TILE[next_tile_idx * 1024 + warpIdx * TILE_ROW + By + j * 32] = B[warpIdx * N + By + j * 32];
        }

        A += 8;
        B += 8 * N;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                A_tmp[k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16 + j) * TILE_COL + k];
            }

            #pragma unroll
            for (int k = 0; k < 4; ++k) {

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    results[j * 4 + k] += A_tmp[t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
                }
            }

        }


        __syncthreads();

        cur_tile_idx = next_tile_idx;
    
    }


    // deal with the last A_TILE and B_TILE.
    #pragma unroll
    for (int j = 0; j < 16; ++j) {

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            A_tmp[k] = A_TILE[cur_tile_idx * 1024 + (warpIdx * 16 + j) * TILE_COL + k];
        }

        #pragma unroll
        for (int k = 0; k < 4; ++k) {

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                results[j * 4 + k] += A_tmp[t] * B_TILE[cur_tile_idx * 1024 + t * TILE_ROW + k * 32 + By];
            }   
        }

    }



    // write result back.
    #pragma unroll
    for (int i = 0; i < 16; ++i) {

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            C[(warpIdx * 16 + i) * N + j * 32 + By] = results[i * 4 + j] * alpha +
                                                      C[(warpIdx * 16 + i) * N + j * 32 + By] * beta;
        }
    }

}