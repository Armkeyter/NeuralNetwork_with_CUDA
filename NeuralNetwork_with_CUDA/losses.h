#pragma once
#include <math.h>

float cross_entropy_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols);
float mse_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols);
float mae_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols);