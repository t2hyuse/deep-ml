#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    // y[i] = a * x[i]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        y[i] = a * x[i];
    }
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    int n = static_cast<int>(x.size());

    if (n==0){
        return{};
    }

    size_t sizebytes = sizeof(float) * static_cast<size_t>(n);

    float* d_x = nullptr;
    cudaMalloc(&d_x, sizebytes);

    cudaMemcpy(d_x, x.data(), sizebytes, cudaMemcpyHostToDevice);

    float* d_y = nullptr;
    cudaMalloc(&d_y, sizebytes);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    scale_kernel<<<blocks, threadsPerBlock>>>(d_x, a, d_y, n);

    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_y, sizebytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return result;
}