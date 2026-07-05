#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "TensorRT header is available." << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "CUDA device count: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    return 0;
}
