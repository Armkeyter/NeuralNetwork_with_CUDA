#include "Activations.h"
#include <math.h>


void sigmoid(float** X, int rows, int cols, bool is_derivative)
{
	if(!is_derivative){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				X[i][j] = 1 / (1 + exp(-X[i][j]));
			}
		}
	}
	else {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				X[i][j] = X[i][j] * (1 - X[i][j]);
			}
		}
	}
}

void tanh(float** X, int rows, int cols, bool is_derivative)
{
}

void relu(float** X, int rows, int cols, bool is_derivative)
{
}

void leakyrelu(float** X, int rows, int cols, bool is_derivative)
{
}

void softmax(float** X, int rows, int cols,float** res, bool is_derivative)
{
	float sum = 0;
	float exp_temp;
	if (!is_derivative) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				sum = 0;
				exp_temp = (float)exp(X[i][j]);
				//sum exponentials
				for (int k = 0; k < cols; k++) {
					sum += (float)exp(X[i][k]);
				}
				res[i][j] = exp_temp / sum;
			}

		}
	}
	else {

	}
}