#include <iostream>

#include "./cuda_kernel.cuh"

#define SHMEM_SIZE = 1024
#define TILE_DIM 16
#define BLOCK_ROWS 8
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

__device__  static void matMul_try(float* data_GPU, float* weights_GPU, float* result,
    int rowsX, int colsX, int colsWeights)
{

    __shared__ float ds_A[TILE_DIM][TILE_DIM];
    __shared__ float ds_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0;

    //double numberoftiles =ceil(m/TILE_WIDTH);

    if (rowsX == colsX == colsWeights) {
        for (int l = 0; l < rowsX / TILE_DIM; ++l) {     //iterate through tiles

            for (int j = 0; j < TILE_DIM; ++j)
                sum += data_GPU[(row * rowsX) + (l * TILE_DIM + j)] * weights_GPU[(l * TILE_DIM + j) * rowsX + col];
            __syncthreads();
        }
        result[row * rowsX + col] = sum;
    }
    else {
        for (int l = 0; l < ceil((float)colsX / TILE_DIM); ++l) {     //iterate through tiles
            if (row < rowsX && l * TILE_DIM + tx < colsX)

                ds_A[ty][tx] = data_GPU[row * colsX + l * TILE_DIM + tx];
            else
                ds_A[ty][tx] = 0.0;

            if (l * TILE_DIM + ty < colsX && col < colsWeights)

                ds_B[ty][tx] = weights_GPU[(l * TILE_DIM + ty) * colsWeights + col];

            else
                ds_B[ty][tx] = 0.0;

            __syncthreads();

            for (int j = 0; j < TILE_DIM && j < colsX; ++j) {    //iterate through elements in the tile

                sum = sum + ds_A[ty][j] * ds_B[j][tx];


            }

            __syncthreads();


        }
        if (row < rowsX && col < colsWeights)

            result[row * colsWeights + col] = sum;
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
// Smart Tile Transpose
__global__ void transposeDiagonal(float* odata, float* idata, int width, int height, const int blockRows)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // diagonal reordering
    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    /*int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;*/
    int xIndex = blockIdx_x * gridDim.x + threadIdx.x;
    int yIndex = blockIdx_y * gridDim.y + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    //printf("%i %i \n", gridDim.x, gridDim.y);
    /*xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;*/
    xIndex = blockIdx_x * gridDim.x + threadIdx.x;
    yIndex = blockIdx_y * gridDim.y + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i = 0; i < TILE_DIM; i += blockRows) {
        tile[threadIdx.y + i][threadIdx.x] =
            idata[index_in + i * width];
        //printf("%i %i %.5f \n", threadIdx.x, threadIdx.y+i, idata[index_in + i * width]);

    }
    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += blockRows)
    {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__device__ void matrix_T_GPU(float* data, float* result, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) 
        result[Col * rows + Row] = data[Row * cols + Col];
}

__device__ void d_weights(float* copy_activation,
    float* prev_activation, float* prev_activation_T, float* result, int X_rows, int X_cols, int W_cols) {

    matrix_T_GPU(prev_activation, prev_activation_T, X_rows, X_cols);
    dev_matmul(prev_activation_T, copy_activation, result, X_cols, X_rows, W_cols);
    //matMul_try(prev_activation_T, copy_activation, result, X_cols, X_rows, W_cols);
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < X_cols) && (Col < W_cols)) {
        result[Row * W_cols + Col] /= X_rows;
    }

}

__device__ void d_biases(float* A, float* B, int rows, int cols) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum += B[i * cols + Col];
        }
        A[Col] = sum / rows;
    }
}


__global__ void forwardProp(float* A, float* B,float* biases, float* C, int rowsA, int colsA, int colsB) {
    dev_matmul(A, B, C, rowsA, colsA, colsB);
    //matMul_try(A, B, C, rowsA, colsA, colsB);
    addbias(C, biases, C, rowsA, colsB);
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

__global__ void count_derivativesCU(float* copy_activation, float* prev_activation,
    float* prev_activation_T, float* result_weights, float* biases, float* result_biases,
    int X_rows, int X_cols, int W_cols) {
    d_biases(biases, result_biases, X_rows, W_cols);
    d_weights(copy_activation, prev_activation, prev_activation_T, result_weights, X_rows, X_cols, W_cols);
}

__global__ void transpose_matmulCU(float* array_to_T, float* result_T, float* data_GPU, float* weights_GPU,
    float* result_matmul, int rowsX, int colsX, int rowsWeights) {

    matrix_T_GPU(array_to_T, result_T, rowsWeights, colsX);
    dev_matmul(data_GPU, weights_GPU, result_matmul, rowsX, colsX, rowsWeights);
    //matMul_try(data_GPU, weights_GPU, result_matmul, rowsX, colsX, rowsWeights);


}

void matrixMultiplication(int threadsN, float* data_GPU, float* weights_GPU, float* result,
    int rowsX, int colsX, int colsWeights) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((colsWeights - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    matmul<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, weights_GPU, result, rowsX, colsX, colsWeights);

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


void forwardPropagation(int threadsN, float* data_GPU, float* weights_GPU,float *biases, float* result,
    int rowsX, int colsX, int colsWeights) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((colsWeights - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }

    forwardProp<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, weights_GPU, biases, result, rowsX, colsX, colsWeights);
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

void transpose_matmul_GPU(int threadsN, float* array_to_T, float* result_T, float* data_GPU, float* weights_GPU, 
    float* result_matmul, int rowsX, int colsX, int rowsWeights)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rowsX - 1) / threadsN + 1, (rowsX - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    transpose_matmulCU << <blocksPerGrid, threadsPerBlock >> > (array_to_T, result_T, data_GPU, 
        weights_GPU,result_matmul, rowsX, colsX, rowsWeights);
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
    //d_weights<<<blocksPerGrid, threadsPerBlock>>>(copy_activation,
    //    prev_activation, prev_activation_T, result, X_rows, X_cols, W_cols);
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
    //d_biases <<<blocksPerGrid, threadsPerBlock >> > (data_GPU, reuslts_GPU, rowsX, colsX);
}

void count_derivatives(int threadsN, float* copy_activation, float* prev_activation, float* prev_activation_T, 
    float* result_weights, float* biases, float* result_biases, int X_rows, int X_cols, int W_cols)
{
    dim3 threadsPerBlock(threadsN);
    dim3 blocksPerGrid((X_rows - 1) / threadsN + 1, (W_cols - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    count_derivativesCU << <blocksPerGrid, threadsPerBlock >> > (copy_activation,prev_activation, prev_activation_T,
        result_weights, biases, result_biases, X_rows, X_cols, W_cols);

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

void try_Transpose(int threadsN,float* input_GPU, float* result_T, int rows, int cols)
{
    std::cout << "Rows: " << rows <<"Cols: " << cols << std::endl;
    dim3 dimGrid;
    

    //}
    //else {
    //    /*if (rows > TILE_DIM)
    //        dimGrid.x = rows / TILE_DIM;
    //    else
    //        dimGrid.x = TILE_DIM / rows;
    //    if (cols > TILE_DIM)
    //        dimGrid.y = cols / TILE_DIM;
    //    else
    //        dimGrid.y = TILE_DIM;
    //        dimGrid.x = TILE_DIM / cols;*/

    //}
    //rows < 32 cols < 32
    //if (rows < TILE_DIM && cols < TILE_DIM) {
    //    dimGrid.x = rows;
    //    dimGrid.y = cols;
    //    dimGrid.z = 1;
    //}
    ////rows > 32 cols > 32
    //else if (rows >= TILE_DIM && cols >= TILE_DIM) {
    /*dimGrid.x = cols / TILE_DIM;
    dimGrid.y = rows / TILE_DIM;
    dimGrid.z = 1;*/
    //}
    //rows 64 cols < 32
    //else if (rows >= TILE_DIM && cols < TILE_DIM) {
    dimGrid.x = 1;
    dimGrid.y = 2;
    dimGrid.z = 1;
    //}

    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    std::cout<<"Num of Blocks " << dimGrid.x << ' ' << dimGrid.y << ' ' << dimGrid.z <<std::endl;
    transposeDiagonal << <dimGrid, dimBlock >> > (result_T, input_GPU, cols, rows, BLOCK_ROWS);
    
}

__global__ void tileMatMull(float* data_GPU, float* weights_GPU, float* result, 
                            int rowsX, int colsX, int colsWeights) {
        
    //check limits( if data is not power of 2(check if thread goes out of the width and height))
    __shared__ float matrix1[TILE_DIM][TILE_DIM];
    __shared__ float matrix2[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int w1_b = colsX * TILE_DIM * by;
    int w1_s = TILE_DIM;
    int w1_e = w1_b + colsX - 1;
    // Index of the first sub-matrix of B processed by the block
    int w2_b = TILE_DIM * bx;

    // Step size used to iterate through the sub-matrices of B
    int w2_s = TILE_DIM * colsWeights;

    float temp=0;


    //#pragma unroll
    for (int i = w1_b, j = w2_b;
        i <= w1_e;
        i += w1_s, j += w2_s) {

        matrix1[ty][tx] = data_GPU[i + colsX * ty + tx];
        matrix2[ty][tx] = weights_GPU[j + colsWeights* ty + tx];
        __syncthreads();
        for (int k = 0; k < TILE_DIM; k++) {
            temp += matrix1[ty][k] * matrix2[k][tx];
        }
        __syncthreads();
    }
    int c = colsWeights * TILE_DIM * by + TILE_DIM * bx;
    result[c + colsWeights * ty + tx] = temp;


    //__shared__ float matrix1[TILE_DIM][TILE_DIM];
    //__shared__ float matrix2[TILE_DIM][TILE_DIM];

    //
    //int tx = threadIdx.x; 
    //int ty = threadIdx.y;

    //int Row = blockIdx.y * TILE_DIM + ty;
    //int Col = blockIdx.x * TILE_DIM + tx;
    //float temp = 0;
    //
    ////#pragma unroll
    //for (int i = 0; i < colsX / TILE_DIM; i++) {
    //    matrix1[ty][tx] = data_GPU[Row * colsX + (i * TILE_DIM + tx)];
    //    matrix2[ty][tx] = weights_GPU[(i*TILE_DIM+ty)*colsX+Col]; /// No Shared Mem Bank conflict
    //    __syncthreads();

    //    for (int j = 0; j < TILE_DIM; j++) {
    //        temp += matrix1[ty][j] * matrix2[j][tx];
    //    }
    //    __syncthreads();
    //    result[Row * colsX + Col] = temp;

    //}

}

void try_MatMull(int threadsN, float* data_GPU, float* weights_GPU, float* result, int rowsX, int colsX, int colsWeights)
{
    //dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid((colsWeights + TILE_DIM - 1) / TILE_DIM, (rowsX + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    //tileMatMull << <dimGrid, dimBlock >> > (data_GPU, weights_GPU, result, rowsX, colsX, colsWeights);
    //matMul_try << <dimGrid, dimBlock >> > (data_GPU, weights_GPU, result, rowsX, colsX, colsWeights);
}

