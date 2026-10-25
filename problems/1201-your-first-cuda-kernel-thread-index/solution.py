#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

std::vector<int> global_thread_indices(int n) {
    if (n <= 0) return {};

    int* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    index_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, n);

    std::vector<int> result(n);
    cudaMemcpy(result.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    return result;
}