#pragma once
#include <cublas_v2.h>

//NOTE: non-template function should follow the rules: declare in header file, difine in source file.

void cublas_sgemm(const float* A, const float* B, float* C, const int N,
                              const float alpha, const float beta);