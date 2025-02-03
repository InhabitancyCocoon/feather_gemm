#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "run_utils.cuh"
#include <string>


int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Too little parameter, you need to chose one kernel.\n");
        return 0;
    }

    constexpr int N = 1 << 11;
    constexpr int bytes = N * N * sizeof(int);
    constexpr float alpha = 1.5, beta = 0.4; // GEMM input parameters, C=α*AB+β*C

    printf("This is a gemm C = alpha * AB + beta * C, use square matrix for simplicity.\n");
    printf("Side length: %d \n", N);


    float *h_A, *h_B, *h_C;
    float *cpu_C; 
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    cpu_C = (float*)malloc(bytes);


    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    random_initialize(h_A, N);
    random_initialize(h_B, N);
    random_initialize(h_C, N);

    copy_matrix(cpu_C, h_C, N);
    cpu_gemm(h_A, h_B, cpu_C, N, alpha, beta);


    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    

    int kernel = std::stoi(argv[1]);

    run_kernel(d_A, d_B, d_C, N, kernel, alpha, beta);

    

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    check_result(cpu_C, h_C, N);




    free(h_A);
    free(h_B);
    free(h_C);
    free(cpu_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}