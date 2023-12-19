#include "./activations.cuh"

//Cross entropy Kernel (Given predictions and labels, returns a matrix containing the cross entropy terms at (i,j))
__global__ void crossentropyCU(float* Y_predict, float* Y, float* loss, int rows, int cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((Row < rows) && (Col < cols)) {
       loss[Row * cols + Col]  = - Y[Row * cols + Col] * logf(Y_predict[Row * cols + Col]);
    }


}

// BEHAVING VERY MYSTERIOUSLY, HANDLE WITH CARE :O
/*
void crossentropy(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols)
{
    float * a;
    a = (float *)malloc(sizeof(float));
    float *d_a;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            a[i * rows + j] = 0;
        }
    cudaMalloc((void **)&d_a, rows*cols*sizeof(float));
    cudaMemcpy(d_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    crossentropyCU<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, Y, d_a, rows, cols);
    cudaDeviceSynchronize();
    
    cudaMemcpy(a, d_a, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    printf("individual cross_entropies\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%f ", a[i * rows + j]);
        printf("\n");
    }
   

    matrixSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, results_GPU, rows, cols);
    cudaDeviceSynchronize();
    free(a);
    cudaFree(d_a);
}
*/

//Result_step is an empty GPU matrix same size as the data, results is a float 
//(Basically wrapper for cross entropies + sum)
void cross_entropy(int threadsN, float* data_GPU, float* Y, float* results_step_GPU,
    float* results_GPU, int rows, int cols){

    cross_entropies(threadsN, data_GPU, Y, results_step_GPU, rows, cols);
    sum( threadsN,  results_step_GPU,  results_GPU,  rows, cols);

}


//Return a matrix of each term of the cross entropy
void cross_entropies(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols){
    dim3 threadsPerBlock(threadsN, threadsN);
    dim3 blocksPerGrid((rows - 1) / threadsN + 1, (rows - 1) / threadsN + 1, 1);
    if (threadsN * threadsN > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(threadsN) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(threadsN) / double(threadsPerBlock.y));
    }
    crossentropyCU<<<blocksPerGrid, threadsPerBlock>>>(data_GPU, Y, results_GPU, rows, cols);
}