# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME : KAWIN K V</h3>
<h3>ENTER YOUR REGISTER NO: 212225040181 </h3>
<h3>EX. NO : 01 </h3>
<h3>DATE</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:

```

%%writefile vector_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)
#define REPEAT 10

// Kernel: Each thread processes two elements
__global__ void sumArraysOnGPU2(float *A, float *B,
                                float *C, int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (i < n)
        C[i] = A[i] + B[i];

    if (i + 1 < n)
        C[i + 1] = A[i + 1] + B[i + 1];
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("GPU-Based Vector Summation\n");
    printf("Device: %s\n", prop.name);
    printf("Vector size: %d\n\n", N);

    size_t bytes = (size_t)N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Initialize arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (float)(i % 100);
        h_B[i] = (float)((i + 1) % 100);
    }

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy input arrays to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Configuration: 256 threads per block
    int threads = 256;
    int gridSize = (N + 2 * threads - 1) / (2 * threads);

    printf("EXPERIMENT 2: Two elements per thread\n");
    printf("Block.x = %d\n", threads);
    printf("Grid.x = %d\n", gridSize);

    // Warm-up
    sumArraysOnGPU2<<<gridSize, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int r = 0; r < REPEAT; r++)
    {
        sumArraysOnGPU2<<<gridSize, threads>>>(d_A, d_B, d_C, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Time elapsed %.6f ms\n", ms / REPEAT);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    int correct = 1;

    for (int i = 0; i < N; i++)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            correct = 0;
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    printf("\nVerification: %s\n",
           correct ? "SUCCESS" : "FAILED");

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    return correct ? 0 : 1;
}
```



## OUTPUT:

<img width="842" height="291" alt="image" src="https://github.com/user-attachments/assets/f90f11c2-3403-40ff-b77f-85148017b00f" />



## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
