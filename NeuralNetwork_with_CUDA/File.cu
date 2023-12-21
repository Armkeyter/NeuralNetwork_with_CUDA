#include <iostream>

#include "./cuda_kernel.cuh"

#define SHMEM_SIZE = 1024

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
__device__ void dev_sigmoid(float* A, float* B, bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = 1 / (1 + exp(-A[Row * cols + Col]));
        else
            B[Row * cols + Col] = A[Row * cols + Col] * (1 - A[Row * cols + Col]);
    }
}

// Very mysterious, I removed the shared memory part because it's late and I'll do it later, but it works ahah
__global__ void matrixSum(float* matrix, float* result, int rows, int cols) {
    // Calculate global thread indices
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize thread-local sum
    float localSum = 0.0f;

    // Check if the thread indices are within the matrix bounds
    if (Row < rows && Col < cols) {
        // Access the element in the matrix
        float element = matrix[Row * cols + Col];

        // Accumulate the element to the local sum
        localSum += element;
    }

    // Perform parallel reduction within the block along the x-dimension
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride && Col + stride < cols) {
            localSum += __shfl_down_sync(0xFFFFFFFF, localSum, stride);
        }
    }

    // Perform parallel reduction within the block along the y-dimension
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.y < stride && Row + stride < rows) {
            localSum += __shfl_down_sync(0xFFFFFFFF, localSum, stride * blockDim.x);
        }
    }

    // The thread with indices (0, 0) in each block writes the result to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(result, localSum);
    }
}
__global__ void hadamardprod(float* data1, float* data2, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        data1[Row * cols + Col] *= data2[Row * cols + Col];
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
__global__ void matrix_T(float* data, float* result, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols))
        result[Col * rows + Row] = data[Row * cols + Col];
}

__device__ void matrix_copy(float* data, float* result, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        result[Row * cols + Col] = data[Row * cols + Col];
    }

}

__device__ void matrix_T_GPU(float* data, float* result, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) 
        result[Col * rows + Row] = data[Row * cols + Col];
}


__global__ void forwardProp(float* A, float* B,float* biases, float* C, int rowsA, int colsA, int colsB) {
    dev_matmul(A, B, C, rowsA, colsA, colsB);
    addbias(C, biases, C, rowsA, colsB);
    //dev_sigmoid(C, C, false, rowsA, colsB);
}



__global__ void softamxCU(float* A, float* B, int rows, int cols,bool is_derivative) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (!is_derivative) {
        if ((Row < rows) && (Col < cols)) {
            B[Row * cols + Col] = (float)expf(A[Row * cols + Col]);
            __syncthreads();

            float sum_exp = 0.0f;
            for (int i = 0; i < cols; i++) {
                sum_exp += B[Row * cols + i];
            }
            B[Row * cols + Col] /= sum_exp;
        }
    }
    else {
        A[Row * cols + Col] -= B[Row * cols + Col];
    }
}

__global__ void d_weights(float* copy_activation,
    float* prev_activation, float* prev_activation_T, float* result, int X_rows, int X_cols, int W_cols) {

    //matrix_copy(activation, copy_matrix, X_rows, X_cols);
    matrix_T_GPU(prev_activation, prev_activation_T, X_rows, X_cols);
    dev_matmul(prev_activation_T, copy_activation, result, X_cols, X_rows, W_cols);
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < X_cols) && (Col < W_cols)) {
        result[Row* W_cols+Col] /= X_rows;
    }

}

__global__ void d_biases(float* A, float* B, int rows, int cols) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum += B[i * cols + Col];
        }
        A[Col] = sum/rows;
    }
}


__global__ void update_weights_kernel(float* weights, float* d_weights, float* biases, float* d_biases, 
                                        float lr,int rowsX, int colsX) {

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rowsX) && (Col < colsX)) {
        weights[Row * colsX + Col] -= lr * d_weights[Row * colsX + Col];
        biases[Col] -= lr * d_biases[Col];
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

//Term by term matrix multiplication
void hadamardproduct(int threadsN, float* data1, float* data2,
    int rows, int cols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((cols - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    hadamardprod << <blocksPerGrid, threadsPerBlock >> > (data1, data2, rows, cols);
}
__global__ void sigmoidCU(float* A, float* B, bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = 1 / (1 + exp(-A[Row * cols + Col]));
        else
            B[Row * cols + Col] = A[Row * cols + Col] * (1 - A[Row * cols + Col]);
    }

}
void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((cols - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    sigmoidCU << <blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, is_derivative, rows, cols);
}




//Computes the sum over a matrix
void sum(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols){
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    matrixSum << <blocksPerGrid, threadsPerBlock >> > (data_GPU, results_GPU, rows, cols);
}
void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    softamxCU << <blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, rows, cols, is_derivative);
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

void matrix_Copy_GPU(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rowsX - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }

    //matrix_copy <<<blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, rowsX, colsX);
}

void matrix_transpose_GPU(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rowsX - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    matrix_T<<<blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, rowsX, colsX);
}

void derivative_weights(int threadsN,float* copy_activation, 
    float* prev_activation, float* prev_activation_T, float* result, int X_rows, int X_cols, int W_cols)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((X_rows - 1) / threadsN + 1, (X_rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    d_weights<<<blocksPerGrid, threadsPerBlock>>>(copy_activation,
        prev_activation, prev_activation_T, result, X_rows, X_cols, W_cols);
}

void derivative_biases(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX)
{
    dim3 threadsPerBlock(threadsN);
    dim3 blocksPerGrid((rowsX - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    d_biases <<<blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, rowsX, colsX);
}

void update_weights_GPU(int threadsN, float* weights, float* d_weights, float* biases, float* d_biases,
                        float lr, int rowsX, int colsX)
{
    dim3 threadsPerBlock(threadsN);
    dim3 blocksPerGrid((rowsX - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    update_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(weights, d_weights, biases, 
        d_biases, lr, rowsX, colsX);
}

