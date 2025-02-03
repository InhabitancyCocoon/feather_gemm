#include "00_cublas.cuh"

void cublas_sgemm(const float* A, const float* B, float* C, const int N,
                              const float alpha, const float beta) {
    cublasStatus_t cbStat;
    cublasHandle_t cbHandle;
    cbStat = cublasCreate(&cbHandle);
    cbStat = cublasSgemm(cbHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                         &alpha, B, N, A, N, &beta, C, N);
}