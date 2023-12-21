#pragma once

//void crossentropy(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols);

void cross_entropy(int threadsN, float* data_GPU, float* Y, float* results_step_GPU, float* results_GPU, int rows, int cols);

void cross_entropies(int threadsN, float* data_GPU, float* Y, float* results_GPU, int rows, int cols);