#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>

cudaError_t addWithCuda(int *c, const int *a, const double *kernel, int n, int m, int kernel_size);


__global__ void addKernel(int *c, const int *a, const double *kernel, int n, int m, int kernel_size)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    int i = x * n + y;
    printf("%d %d\n", x, y);
    int row = i / m;
    int col = i % m;
    double c_double = 0.0;
    for (int j = 0; j < kernel_size; j++) {
        for (int k = 0; k < kernel_size; k++) {
            int center_element = kernel_size / 2;
            int row_center_element = j - center_element;
            int col_center_element = k - center_element;
            c_double +=
                0 <= row + row_center_element && row + row_center_element < n &&
                0 <= col + col_center_element && col + col_center_element < m ?
                (1.0 * a[i + m * row_center_element + col_center_element]) * kernel[j * kernel_size + k] : 0.0;

        }
    }

    c[i] = c_double;
}

int main()
{
    int n, m, kernel_size = 3;
    std::ifstream f("image_hard.in");
    f >> n >> m;

    int* a = (int*)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++) {
        f >> a[i];
    }
    double* kernel = (double*)malloc(kernel_size * kernel_size * sizeof(double));
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = 1.0 / 9.0;
    }
    int *c = (int*)malloc(n * m * sizeof(int));

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, kernel, n, m, kernel_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", c[i * m + j]);
        }
        printf("\n");
    }
    free(a);
    free(kernel);
    free(c);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const double *kernel, int n, int m, int kernel_size)
{
    int *dev_a = 0;
    double *dev_kernel = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, n * m * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, n * m * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_kernel, kernel_size * kernel_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_kernel, kernel, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.

    addKernel<<<n, m>>>(dev_c, dev_a, dev_kernel, n, m, kernel_size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_kernel);
    
    return cudaStatus;
}
