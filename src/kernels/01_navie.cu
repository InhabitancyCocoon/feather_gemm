


__global__ void  naive_matmul(const float* A, const float* B, float* C, const int N,
                              const float alpha, const float beta) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float value = C[x * N + y] * beta;
    if (x < N && y < N) {
        for (int k = 0; k < N; k++) {
            value += A[x * N + k] * B[k * N + y] * alpha;
        }
    }
    C[x * N + y] = value;
}