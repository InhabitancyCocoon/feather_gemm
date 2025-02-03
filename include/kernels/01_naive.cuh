#pragma once

__global__ void  naive_matmul(const float* A, const float* B, float* C, const int N,
                              const float alpha, const float beta);