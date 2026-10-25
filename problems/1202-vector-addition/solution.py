#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    int n = static_cast<int>(a.size());
    if (n == 0) {
        return {};
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> c(n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // cudaMemcpy on the default stream synchronizes with the kernel launch
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}