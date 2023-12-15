#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void matrixMultiplication(int threadsN, float* data_GPU, float* weights_GPU, float* result,
	int rowsX, int colsX, int rowsWeights);


void forwardPropagation(int threadsN, float* data_GPU, float* weights_GPU, float* biases, float* result,
	int rowsX, int colsX, int rowsWeights);

void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX, bool is_derivative = false);

void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);

void getDeviceInfo();