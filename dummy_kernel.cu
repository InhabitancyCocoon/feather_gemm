#include <stdio.h>
#include <run_utils.cuh>

__global__ void times_2(int *data, const int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2;
}


int main(int argc, char* argv[]) {

    simple_device_query();


    const int N = 1 << 8;
    dim3 grid(4);
    dim3 block(N / 4);
    int* h_data = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) *(h_data + i) = i;
    int* d_data;
    cudaMalloc((void**)&d_data, sizeof(int) * N);
    cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);
    times_2<<<grid,block>>>(d_data, N);
    cudaMemcpy(h_data, d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    bool flag = true;
    for (int i = 0; i < N; ++i) {
        if (*(h_data + i) != i << 1) {
            flag = false;
            break;
        }
    }

    printf("%s", flag ? "right" : "wrong");

    cudaFree(h_data);
    free(h_data);
    return 0;

}