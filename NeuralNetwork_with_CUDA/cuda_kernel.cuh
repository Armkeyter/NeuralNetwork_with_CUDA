#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void hadamardproduct(int threadsN, float* data1, float* data2, float* result,
    int rows, int cols);

void matrixMultiplication(int threadsN, float* data_GPU, float* weights_GPU, float* result,
	int rowsX, int colsX, int rowsWeights);


void forwardPropagation(int threadsN, float* data_GPU, float* weights_GPU, float* biases, float* result,
	int rowsX, int colsX, int rowsWeights);

void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX, bool is_derivative = false);

void tanh(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

void relu(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

void leakyrelu(int threadsN, float alpha, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

//void crossentropy(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols);

void cross_entropy(int threadsN, float* data_GPU, float* Y, float* results_step_GPU, float* results_GPU, int rows, int cols);

void cross_entropies(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols);

void sum(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols);

void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);

void getDeviceInfo();