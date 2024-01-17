#include "./activations.cuh"

__global__ void tanh(float* A, float* B,bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = 2 / (1 + exp(-2 * A[Row * cols + Col])) - 1;
        else
            B[Row * cols + Col]  = 1 - pow(A[Row * cols + Col], 2);
    }
    
}

__global__ void relu(float* A, float* B,bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = (A[Row * cols + Col] >= 0) ? A[Row * cols + Col] : 0;
        else
            B[Row * cols + Col] = (A[Row * cols + Col] >= 0) ? 1 : 0;
    }
    
}

__global__ void leakyrelu(float alpha, float* A, float* B,bool is_derivative, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < rows) && (Col < cols)) {
        if (!is_derivative)
            B[Row * cols + Col] = (A[Row * cols + Col] >= 0) ? A[Row * cols + Col] : alpha* A[Row * cols + Col];
        else
            B[Row * cols + Col] = (A[Row * cols + Col] >= 0) ? 1 : alpha;
    }
    
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


__global__ void softamxCU(float* A, float* B, int rows, int cols, bool is_derivative) {
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

void tanh(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    tanh<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, results_GPU, is_derivative, rows, cols);
}

void relu(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    relu<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, results_GPU, is_derivative, rows, cols);
}

void leakyrelu(int threadsN, float alpha, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative)
{
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    leakyrelu<<<blocksPerGrid, threadsPerBlock>>>(alpha, data_GPU, results_GPU, is_derivative, rows, cols);
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

