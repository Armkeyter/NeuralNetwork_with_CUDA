#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void sigmoid(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX, bool is_derivative = false);

void tanh(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

void relu(int threadsN, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

void leakyrelu(int threadsN, float alpha, float* data_GPU, float* results_GPU, int rows, int cols, bool is_derivative);

void softmax(int threadsN, float* data_GPU, float* reuslts_GPU, int rowsX, int colsX);

