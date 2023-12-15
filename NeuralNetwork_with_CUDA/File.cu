#include <iostream>

#include "./cuda_kernel.cuh"



void getDeviceInfo() {
    const int kb = 1024;
    const int mb = kb * kb;
    int device;
    cudaGetDeviceCount(&device);
    cudaDeviceProp props;
    if (device < 1) {
        std::cout << "No GPU was found" << std::endl;
    }
    else {
        cudaGetDeviceProperties(&props, 0);
        std::cout << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "Warp Size: \t\t" << props.warpSize << std::endl;
        std::cout << "Threads Per Block: \t" << props.maxThreadsPerBlock << std::endl;
        std::cout << "Max Block dim: \t\t" << props.maxThreadsDim[0] << ',' << props.maxThreadsDim[1] << ',' << props.maxThreadsDim[2] << std::endl;
        std::cout << "Max Grid dim: \t\t" << props.maxGridSize[0] << ',' << props.maxGridSize[1] << ',' << props.maxGridSize[2] << std::endl;
        std::cout << "Global Memory: \t\t" << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "Shared Memory: \t\t" << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "Constant Memory: \t" << props.totalConstMem / kb << "kb" << std::endl;
        std::cout << "Block Registers: \t" << props.regsPerBlock << std::endl << std::endl;
    }
}


__device__ void dev_matmul(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < rowsA) && (Col < colsB)) {
        float Cvalue = 0;
        for (int k = 0; k < colsA; ++k)
            Cvalue += A[Row * colsA + k] * B[k * colsB + Col];
        C[Row * colsB + Col] = Cvalue;
    }
}

__device__ void addbias(float* data, float* biases,float* result, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        result[Row * cols + Col] = data[Row * cols + Col] + biases[Col];
    }
    
}


__global__ void matmul(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < rowsA) && (Col < colsB)) {
        float Cvalue = 0;
        for (int k = 0; k < colsA; ++k)
            Cvalue += A[Row * colsA + k] * B[k * colsB + Col];
        C[Row * colsB + Col] = Cvalue;
    }
}


__global__ void forwardProp(float* A, float* B,float* biases, float* C, int rowsA, int colsA, int colsB) {
    dev_matmul(A, B, C, rowsA, colsA, colsB);
    addbias(C, biases, C, rowsA, colsB);
}

__global__ void sigmoidCU(float* A, float* B,bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = 1 / (1 + exp(-A[Row * cols + Col]));
        else
            B[Row * cols + Col] = A[Row * cols + Col] * (1 - A[Row * cols + Col]);
    }
}

__global__ void softamxCU(float* A, float* B, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        B[Row * cols + Col] = 1 / (1 + exp(-A[Row * cols + Col]));
    }
}

void matrixMultiplication(int threadsN, float* data_GPU, float* weights_GPU, float* result,
    int rowsX, int colsX, int rowsWeights) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rowsWeights - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    matmul<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, weights_GPU, result, rowsX, colsX, rowsWeights);
}

void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    sigmoidCU<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, reuslts_GPU, is_derivative, rows, cols);
}

void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rows, int cols)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    softamxCU<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, reuslts_GPU, rows, cols);
}


void forwardPropagation(int threadsN, float* data_GPU, float* weights_GPU,float *biases, float* result,
    int rowsX, int colsX, int rowsWeights) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rowsWeights - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }

    forwardProp<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, weights_GPU, biases, result, rowsX, colsX, rowsWeights);
}
