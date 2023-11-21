#include "Neural_Network.h"
#include "./activations.h"
#include <random>

float Neural_Network::dot_product(float* X, float* W,int size)
{
	float res = 0.0f;
	for (int i = 0; i < size; i++)
		res += X[i] * W[i];

	return res;
}

void Neural_Network::get_column(float** W, int row_size, int index,float result[])
{
	for (int i = 0; i < row_size; ++i)
		result[i] = W[i][index];
}

void Neural_Network::matrix_multiplication(float** X, float** W, float** res, int row_size, int col_size, int W_col_size)
{
	float* column = new float[col_size];
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < W_col_size; j++) {
			get_column(W, col_size, j, column);
			res[i][j] = dot_product(X[i],column,col_size);
		}
	}
	delete[] column;
}

Neural_Network::Neural_Network(int* architecture,int size)
{
	//If array is empty
	if (architecture[0] == 0 or size==0) {
		throw std::invalid_argument("size of architecture is 0");
	}
	
	architecture_length = size;
	this->architecture = new int[architecture_length+1];
	array_weights = new float** [architecture_length];
	array_biases = new float* [architecture_length];
	//Creating second dimension of arrays weights and biases
	for (int i = 0; i < architecture_length; i++) {
		array_weights[i] = new float* [architecture[i]];
		array_biases[i] = new float[architecture[i+1]];
		//Initializing 3 dimension of weights array
		for (int j = 0; j < architecture[i]; j++) {
			array_weights[i][j] = new float[architecture[i + 1]];
		}
	}
	for (int i = 0; i < architecture_length+1; ++i) {
		this->architecture[i] = architecture[i];
	}
	init_weights();
}

Neural_Network::~Neural_Network()
{
	// Deleting weights, biases
	for (int i = 0; i < architecture_length; i++) {

		for (int j = 0; j < architecture[i];j++) {
 			delete array_weights[i][j];
		}
		delete[] array_weights[i];
		delete[] array_biases[i];
	}
	delete[] architecture;

	delete[] array_weights;
	delete[] array_biases;
}

void Neural_Network::init_weights()
{
	// Set up a random number generator
	//std::random_device rd;
	//std::mt19937 gen(rd());
	std::mt19937 gen(1);
	std::uniform_real_distribution<float> dis(-0.5, 0.5);
	// Generate a random number between -0.5 and 0.5
	float randomValue = dis(gen);
	//Each weight,bias entity
	for (int i = 0; i < architecture_length; i++) {
		//For each row
		for (int j = 0; j < architecture[i]; j++) {
			//For each column
			for (int k = 0; k < architecture[i+1]; k++) {
				array_weights[i][j][k] = dis(gen);
			}
		}
	}
	// init biases
	for (int i = 0; i < architecture_length; i++)
		for(int j=0;j<architecture[i+1];j++)
			array_biases[i][j] = dis(gen);
}

void Neural_Network::print_weights_biases()
{
	std::cout << "WEIGHTS" << std::endl;
	for (int i = 0; i < architecture_length; i++) {
		//For each row
		for (int j = 0; j < architecture[i]; j++) {
			//For each column
			for (int k = 0; k < architecture[i+1]; k++) {
				std::cout << array_weights[i][j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}
	std::cout << "BIASES" << std::endl;
	for (int i = 0; i < architecture_length; i++) {
		for (int j = 0; j < architecture[i + 1]; j++)
			std::cout << array_biases[i][j] << ' ';
		std::cout << std::endl;
	}


}

void Neural_Network::forward_propagation(float** X, float** W, float* b,float** res,int row_size,int col_size,int W_col_size)
{
	matrix_multiplication(X, W, res, row_size, col_size, W_col_size);
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < W_col_size; j++) {
			res[i][j] += b[j];
		}
	}
}

void Neural_Network::fit(float** X, float* Y,int X_rows,int X_cols)
{
	// Check if the size of input data the same as the input layer of the NN
	if (X_cols != architecture[0]) {
		throw std::invalid_argument("Size of matrix X doesn't match size of matrix W (X.row != W.column)");
	}
	float*** Z = new float** [architecture_length+1];
	
	// init first result XW weights
	Z[0] = new float* [X_rows];
	for (int j = 0; j < X_rows; j++)
		Z[0][j] = new float[architecture[1]];

	forward_propagation(X, array_weights[0], array_biases[0], Z[0], X_rows, X_cols, architecture[1]);
	if (architecture_length >= 2) {
		sigmoid(Z[0], X_rows, architecture[1]);
		
		for (int i = 1; i < architecture_length; i++) {
			Z[i] = new float* [X_rows];
			for (int j = 0; j < X_rows; j++)
				Z[i][j] = new float[architecture[i+1]];
			forward_propagation(Z[i-1], array_weights[i], array_biases[i], Z[i], X_rows, architecture[i], architecture[i+1]);
			//if a hidden layer - do activation function
			if(i!=architecture_length-1)
				sigmoid(Z[i], X_rows, architecture[i+1]);
		}
	}

	Z[architecture_length] = new float* [X_rows];
	for (int i = 0; i < X_rows; i++) {
		Z[architecture_length][i] = new float[architecture[architecture_length]];
	}
	softmax(Z[architecture_length-1], X_rows, architecture[architecture_length], Z[architecture_length]);

	for (int i = 0; i < architecture_length; i++) {
		std::cout << "Weights: " << i << std::endl;
		for (int j = 0; j < X_rows; j++) {
			for (int k = 0; k < architecture[i+1]; k++) {
				std::cout << Z[i][j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}
	std::cout << "Softmax: " << std::endl;
	for (int j = 0; j < X_rows; j++) {
		for (int k = 0; k < architecture[architecture_length]; k++) {
			std::cout << Z[architecture_length][j][k] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;




	for (int i = 0; i < architecture_length+1; i++) {

		for (int j = 0; j < X_rows; j++) {
			delete Z[i][j];
		}
		delete[] Z[i];
	}
	delete[] Z;
}
