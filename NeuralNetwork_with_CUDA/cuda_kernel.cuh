#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void hadamardproduct(int threadsN, float* data1, float* data2,
	int rows, int cols);

void matrixMultiplication(int threadsN, float* data_GPU, float* weights_GPU, float* result,
	int rowsX, int colsX, int rowsWeights);


void forwardPropagation(int threadsN, float* data_GPU, float* weights_GPU, float* biases, float* result,
	int rowsX, int colsX, int rowsWeights);

void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX, bool is_derivative = false);

void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX, bool is_derivative = false);

void getDeviceInfo();

void matrix_Copy_GPU(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);
void matrix_transpose_GPU(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);

void derivative_weights(int threadsN, float* copy_activation,
	float* prev_activation, float* prev_activation_T, float* result,
	int X_rows, int X_cols, int W_cols);

void derivative_biases(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);

void update_weights_GPU(int threadsN, float* weights, float* d_weights, float* biases, 
						float* d_biases, float lr, int rowsX, int colsX);