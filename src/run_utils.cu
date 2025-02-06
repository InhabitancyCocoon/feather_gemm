#include "run_utils.cuh"


void random_initialize(float *data, const int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            data[i * N + j] = static_cast<float>(rand() & 0xFF) / 200.0f;
        }
    }
}


void check_result(const float* data_1, const float* data_2, const int N) {
    constexpr float eps = 1e-2;
    bool consistent = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float diff = abs(data_1[i * N + j] - data_2[i * N + j]);
            if (diff > eps) {
                printf("the kernel is wrong! diff at [%d, %d] abs(%.2f - %.2f) = %.2f \n", i, j, data_1[i * N + j], data_2[i * N + j], diff);
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            break;
        }
    }
    if (consistent) {
        printf("Congratulations! your kernel works well! \n");
    }
}


__global__ void warm_up_kernel(float* C, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = C[idx];
    C[idx] = value;
}


void run_kernel(float* A, float* B, float* C, const int N, const int kernel,
                const float alpha, const float beta) {
    constexpr unsigned int WARP_SIZE = 32;
    dim3 grid;
    dim3 block;


    warm_up_kernel<<<256, N / 256>>>(C, N);
    cudaDeviceSynchronize();

    printf("Warm up kernel finished!\n");


    switch (kernel)
    {
    case 0:
        printf("Use default cuBlas kernel.\n");
        cublas_sgemm(A, B, C, N, alpha, beta);
        break;
    
    case 1:
        printf("Use naive matmul kernel.\n");
        block = dim3(32, 32);
        grid = dim3(N / 32, N / 32);
        
        naive_matmul<<<grid,block>>>(A, B, C, N, alpha, beta);
        
        break;

    case 2:
        printf("Use coalesce memory access kernel.\n");
        
        block = dim3(WARP_SIZE * WARP_SIZE);
        grid = dim3(N / WARP_SIZE, N / WARP_SIZE);
        coalesce_matmul<WARP_SIZE><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 3:
        printf("Tiled matmul.\n");
        block = dim3(WARP_SIZE * WARP_SIZE);
        grid = dim3(N / WARP_SIZE, N / WARP_SIZE);
        tiled_matmul<WARP_SIZE><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 4:
        printf("4x coarsened tiled matmul.\n");
        block = dim3(WARP_SIZE * WARP_SIZE);
        grid = dim3(N / WARP_SIZE / 2, N / WARP_SIZE / 2);
        tiled_coarsened_matul_4x<WARP_SIZE, 2><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 5:
        //Note: be careful of the launch configuration when coarsening.
        printf("One thread computes 8 elements in C.\n");
        block = dim3(64 * 8);
        grid = dim3(N / 64, N / 64);
        block_tiling_1D_matmul<64,8,8><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 6:
        printf("One thread now computes 8 x 8 elements in C.\n");
        block = dim3(16 * 16);
        grid = dim3(N / 128, N / 128);
        block_tiling_2D_matmul<128, 8, 4, 8><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 7:
        printf("Use vectorize memory access and transpose smem of A_TILE.\n");
        block = dim3(16 * 16);
        grid = dim3(N / 128, N / 128);
        vectorize<128, 8, 4, 8><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;

    case 8:
        printf("Warp level gemm.\n");
        block = dim3(16 * 16);
        grid = dim3(N / 128, N / 128);
        warp_tile<128, 8><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;


    case 9:
        printf("Double buffering.\n");
        block = dim3(16 * 16);
        grid = dim3(N / 128, N / 128);
        double_buffering<128, 8><<<grid,block>>>(A, B, C, N, alpha, beta);
        break;
    

    default:
        printf("Invalid kernel!\n");
        break;
    }

}


void cpu_gemm(const float* A, const float* B, float* C, const int N, const float alpha, const float beta) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = beta * C[i * N + j];
            for (int k = 0; k < N; ++k) {
                value += alpha * A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = value;
        }
    }
}


void simple_device_query() {

    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDevice(&iDev);
    cudaGetDeviceProperties(&iProp, iDev);

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors:  %d \n",
            iProp.multiProcessorCount);
    printf("Total amount of global memory: %4.2f GB \n",
            iProp.totalGlobalMem / (1024 * 1024 * 1024.0));
    printf("Total amount of constant memory:  %4.2f KB \n",
            iProp.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block:  %4.2f KB\n",
            iProp.sharedMemPerBlock / 1024.0);

    printf("Total amount of shared memory per multiprocessor:  %4.2f KB \n",
            iProp.sharedMemPerMultiprocessor / 1024.0);
    printf("Total number of registers available per block: %d\n",
            iProp.regsPerBlock);

    printf("Maximum number of registers per thread can use: %d\n",
            255);

    printf("Totoal number of registers available per SM: %d\n",
            iProp.regsPerMultiprocessor);
    printf("Warp size:                                     %d\n",
            iProp.warpSize);
    printf("Maximum number of threads per block:           %d\n",
            iProp.maxThreadsPerBlock);
    printf("Maximum blocks Per MultiProcessor: %d\n",
            iProp.maxBlocksPerMultiProcessor);
    printf("Maximum number of threads per multiprocessor:  %d\n",
            iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:    %d\n",
            iProp.maxThreadsPerMultiProcessor / iProp.warpSize);

    printf("local L1 cache supported: %s \n",
            iProp.localL1CacheSupported ? "true" :"false");


    printf("l2 cache szie %d MB \n", 
            iProp.l2CacheSize / (1024 * 1024));

    printf("persisting l2 cache max size: %d MB \n",
            iProp.persistingL2CacheMaxSize / (1024 * 1024));

    printf("Device can possibly execute multiple kernels concurrently: %d \n", iProp.concurrentKernels);

    

    
    int major, minor;


    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, iDev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, iDev);

    printf("Compute capability: %d %d \n", major, minor);
}


void copy_matrix(float* dst, const float* src, const int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dst[i * N + j] = src[i * N + j];
        }
    }
}


