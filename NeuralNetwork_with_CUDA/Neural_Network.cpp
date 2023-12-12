#include "Neural_Network.h"
#include "./activations.h"
#include "./losses.h"
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./cuda_kernel.cuh"

//#define DEBUG 

float** Neural_Network::matrix_copy(float** X, int row_size, int col_size) {
	float** res = new float* [row_size];
	for (int i = 0; i < row_size; i++) {
		res[i] = new float[col_size];
		for (int j = 0; j < col_size; j++) {
			res[i][j] = X[i][j];
		}
	}
	return res;
}

float Neural_Network::dot_product(float* X, float* W,int size)
{
	float res = 0.0f;
	for (int i = 0; i < size; i++)
		res += X[i] * W[i];

	return res;
}

float** Neural_Network::minus_matrix_return(float** X, float** Y, int row_size, int col_size) {
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			X[i][j] = X[i][j] - Y[i][j];
		}
	}
	return X;
}

void Neural_Network::get_column(float** W, int row_size, int index,float result[])
{
	for (int i = 0; i < row_size; ++i)
		result[i] = W[i][index];
}

void Neural_Network::minus_matrix(float** X, float** Y, int row_size, int col_size) {
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			X[i][j] -= Y[i][j];
		}
	}
}

float* Neural_Network::minus_vector_return(float* X, float* Y, int row_size) {
	for (int i = 0; i < row_size; i++) {
		X[i] = X[i] - Y[i];
	}
	return X;
}

void Neural_Network::hadamard(float** X, float** Y, int row_size, int col_size) {
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			X[i][j] *= Y[i][j];
		}
	}
}

float** Neural_Network::hadamard_return(float** X, float** Y, int row_size, int col_size) {
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			X[i][j] = X[i][j] * Y[i][j];
		}
	}
	return X;
}

float** Neural_Network::scalar_multiply_return(float** X, float* y, int row_size, int col_size) {
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			X[i][j] = *y * X[i][j];
		}
	}
	return X;
}

float* Neural_Network::scalar_multiply_return(float* X, float* y, int row_size) {
	for (int i = 0; i < row_size; i++) {

		X[i] = *y * X[i];
	}
	return X;
}

float** Neural_Network::matrix_transpose(float** X, int row_size, int col_size) {
	float** res = new float* [col_size];
	for (int i = 0; i < col_size; i++) {
		res[i] = new float[row_size];
	}
	for (int i = 0; i < col_size; i++) {
		for (int j = 0; j < row_size; j++) {
			res[i][j] = X[j][i];
		}
	}
	return res;
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

float** Neural_Network::matrix_multiplication_return(float** X, float** W, int row_size, int col_size, int W_col_size)
{
	float** res = new float* [row_size];
	for (int i = 0; i < row_size; i++) {
		res[i] = new float[W_col_size];
	}
	float* column = new float[col_size];
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < W_col_size; j++) {
			get_column(W, col_size, j, column);
			res[i][j] = dot_product(X[i], column, col_size);
		}
	}
	delete[] column;
	return res;
}

void Neural_Network::update_weights(float lr)
{
	for (int k = 0; k < a_len; k++) {
		for (int i = 0; i < architecture[k]; i++) {
			for (int j = 0; j < architecture[k + 1]; j++) {
				array_weights[k][i][j] -= lr * dW[a_len - 1 - k][i][j];
			}
		}
	}
	for (int i = 0; i < a_len; i++) {
		for (int j = 0; j < architecture[i + 1]; j++) {
			array_biases[i][j] -= lr * db[a_len - 1 - i][j];
		}
	}
}

void Neural_Network::back_propagation(float** X, int** Y, float*** Z, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < architecture[a_len]; j++)
			Z[a_len - 1][i][j] -= Y[i][j];

	#ifdef DEBUG
		std::cout << "Softmax-Y " << std::endl;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < architecture[a_len]; j++) {
				std::cout << Z[a_len - 1][i][j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;

	#endif // DEBUG


	float sum = 0.0f;
	//transposed sigmoid activation
	float** delta_T = matrix_copy(Z[a_len - 1], rows, cols);
	for (int i = 0; i < a_len - 1; i++) {
		float** a_prev = matrix_transpose(Z[a_len - i - 2], rows, architecture[a_len - i - 1]);

		//DW
		matrix_multiplication(a_prev, delta_T, dW[i], architecture[a_len - i - 1],
			rows, architecture[a_len - i]);

		for (int j = 0; j < architecture[a_len - i - 1]; j++) {
			delete[] a_prev[j];
		}
		delete[] a_prev;

		// DW/N
		for (int j = 0; j < architecture[a_len - i - 1]; j++)
			for (int k = 0; k < architecture[a_len - i]; k++)
				dW[i][j][k] /= rows;
		//DB
		for (int j = 0; j < architecture[a_len - i]; j++) {
			sum = 0.0f;
			for (int k = 0; k < rows; k++) {
				sum += delta_T[k][j];
			}
			db[i][j] = sum / rows;
		}
	#ifdef DEBUG

			std::cout << "DW " << i << ':' << std::endl;
			for (int j = 0; j < architecture[a_len - i - 1]; j++) {
				for (int k = 0; k < architecture[a_len - i]; k++) {
					std::cout << dW[i][j][k] << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << "Db " << i << ':' << std::endl;
			for (int j = 0; j < architecture[a_len - i]; j++)
				std::cout << db[i][j] << ' ';
			std::cout << std::endl;
			std::cout << std::endl;
	#endif // DEBUG

		//DZ
		sigmoid(Z[a_len - i - 2], rows, architecture[a_len - i - 1], true);
		// W.T
		float** w_t = matrix_transpose(array_weights[a_len - i - 1], architecture[a_len - i - 1], architecture[a_len - i]);

		// np.dot(softmax,weights[a_len-1].T)
		float** temp = new float* [rows];
		for (int j = 0; j < rows; j++) {
			temp[j] = new float[architecture[a_len - i - 1]];
		}
		matrix_multiplication(delta_T, w_t, temp, rows, architecture[a_len - i], architecture[a_len - i - 1]);
		// np.multiply(np.dot(delta_T, w2.T), der_sigmoid(sig))
		hadamard(temp, Z[a_len - i - 2], rows, architecture[a_len - i - 1]);

		for (int j = 0; j < rows; j++) {
			delete[] delta_T[j];
		}
		for (int j = 0; j < rows; j++) {
			delta_T[j] = new float[architecture[a_len - i - 1]];
			for (int k = 0; k < architecture[a_len - i - 1]; k++) {
				delta_T[j][k] = temp[j][k];
			}
		}
		#ifdef DEBUG
				std::cout << "dZ " << i << ':' << std::endl;
				for (int j = 0; j < rows; j++) {
					for (int k = 0; k < architecture[a_len - i - 1]; k++) {
						std::cout << delta_T[j][k] << ' ';
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				std::cout << std::endl;
		#endif // DEBUG
		for (int j = 0; j < architecture[a_len - i]; j++) {
			delete[] w_t[j];
		}
		delete[] w_t;
		for (int j = 0; j < rows; j++) {
			delete[] temp[j];
		}
		delete[] temp;
	}
	float** X_T = matrix_transpose(X, rows, cols);
	matrix_multiplication(X_T, delta_T, dW[a_len - 1], cols, rows, architecture[1]);

	//// dW / N - num of examples
	for (int i = 0; i < cols; i++)
		for (int j = 0; j < architecture[1]; j++)
			dW[a_len - 1][i][j] /= rows;

	//Count db last
	for (int i = 0; i < architecture[1]; i++) {
		sum = 0.0f;
		for (int j = 0; j < rows; j++) {
			sum += delta_T[j][i];
		}
		db[a_len - 1][i] = sum / rows;
	}
	#ifdef DEBUG
		std::cout << "DW_Last: " << std::endl;
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < architecture[1]; k++) {
				std::cout << dW[a_len - 1][j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "Db_Last: " << std::endl;
		for (int i = 0; i < architecture[1]; i++) {
			std::cout << db[a_len - 1][i] << ' ';
		}
		std::cout << std::endl;
		std::cout << std::endl;
	#endif // DEBUG
	for (int i = 0; i < rows; i++) {
		delete[] delta_T[i];
	}
	delete[] delta_T;

	for (int i = 0; i < cols; i++) {
		delete[] X_T[i];
	}
	delete[] X_T;

}

Neural_Network::Neural_Network(int* architecture,int size,int seed)
{
	//If array is empty
	if (architecture[0] == 0 or size==0) {
		throw std::invalid_argument("size of architecture is 0");
	}
	
	a_len = size;
	this->architecture = new int[a_len+1];
	array_weights = new float** [a_len];
	array_biases = new float* [a_len];
	dW = new float** [a_len];
	db = new float* [a_len];
	//Creating second dimension of arrays weights and biases
	for (int i = 0; i < a_len; i++) {
		array_weights[i] = new float* [architecture[i]];
		array_biases[i] = new float[architecture[i+1]];
		// reverse order because we will do back prop
		dW[i] = new float* [architecture[a_len-i-1]];
		db[i] = new float [architecture[a_len - i]];
		//Initializing 3 dimension of weights array
		for (int j = 0; j < architecture[i]; j++) {
			array_weights[i][j] = new float[architecture[i + 1]];
			
		}
		for (int j = 0; j < architecture[a_len - i-1]; j++) {
			dW[i][j] = new float[architecture[a_len - i]];
		}
	}
	for (int i = 0; i < a_len+1; ++i) {
		this->architecture[i] = architecture[i];
	}
	init_weights(seed);
}

Neural_Network::~Neural_Network()
{
	// Deleting weights, biases
	for (int i = 0; i < a_len; i++) {

		for (int j = 0; j < architecture[i];j++) {
 			delete array_weights[i][j];
		}
		// Opossite direction
		for (int j = 0; j < architecture[a_len - i-1]; j++) {
			delete dW[i][j];
		}
		delete[] array_weights[i];
		delete[] dW[i];
		delete[] array_biases[i];
		delete[] db[i];
	}
	delete[] architecture;
	delete[] array_weights;
	delete[] dW;
	delete[] array_biases;
	delete[] db;
}

void Neural_Network::init_weights(unsigned int seed)
{
	// Set up a random number generator
	//std::random_device rd;
	//std::mt19937 gen(rd());
	//std::mt19937 gen;
	std::random_device rd;
	unsigned int actual_seed = (seed != 0) ? seed : rd();
	std::mt19937 gen(actual_seed);
	
	std::uniform_real_distribution<float> dis(-0.5, 0.5);
	// Generate a random number between -0.5 and 0.5
	float randomValue = dis(gen);
	//Each weight,bias entity
	for (int i = 0; i < a_len; i++) {
		//For each row
		for (int j = 0; j < architecture[i]; j++) {
			//For each column
			for (int k = 0; k < architecture[i+1]; k++) {
				array_weights[i][j][k] = dis(gen);
			}
		}
	}
	// init biases
	for (int i = 0; i < a_len; i++)
		for(int j=0;j<architecture[i+1];j++)
			array_biases[i][j] = dis(gen);
}

void Neural_Network::print_weights_biases()
{
	std::cout << "WEIGHTS" << std::endl;
	for (int i = 0; i < a_len; i++) {
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
	for (int i = 0; i < a_len; i++) {
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

void Neural_Network::fit(float** X, int** Y,int X_rows,int X_cols,int epochs,float lr)
{
	// Check if the size of input data the same as the input layer of the NN
	if (X_cols != architecture[0]) {
		throw std::invalid_argument("Size of matrix X doesn't match size of matrix W (X.row != W.column)");
	}
	float*** Z = new float** [a_len];
	
	// init first result XW weights
	Z[0] = new float* [X_rows];
	for (int j = 0; j < X_rows; j++)
		Z[0][j] = new float[architecture[1]];

	#ifdef GPU

		matrix_multiplication(X, array_weights[0], Z[0], X_rows, X_cols, architecture[1]);
		for (int i = 0; i < X_rows; i++) {
			for (int j = 0; j < architecture[1]; j++) {
				Z[0][i][j] += array_biases[0][j];
			}
		}

		std::cout << "XW+b: " << std::endl;
		for (int j = 0; j < X_rows; j++) {
			for (int k = 0; k < architecture[1]; k++) {
				std::cout << Z[0][j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		// FORWARD FUNCTION
		float* flattenedX = new float[X_rows * X_cols];
		float* flattenWeights = new float[architecture[0] * architecture[1]];
		float* result = new float[X_rows * architecture[1]];

		for (int i = 0; i < X_rows; ++i)
			for (int j = 0; j < X_cols; ++j)
				flattenedX[i * X_cols + j] = X[i][j];

		for (int i = 0; i < architecture[0]; i++)
			for (int j = 0; j < architecture[1]; j++)
				flattenWeights[i * architecture[1] + j] = array_weights[0][i][j];

		float* result_GPU, * x_GPU, * weights_GPU, * biases_GPU;
		
		cudaMalloc((void**)&x_GPU, X_rows * X_cols * sizeof(float));
		cudaMalloc((void**)&weights_GPU, architecture[0] * architecture[1] * sizeof(float));
		cudaMalloc((void**)&biases_GPU, architecture[1] * sizeof(float));
		cudaMalloc((void**)&result_GPU, X_rows* architecture[1] * sizeof(float));

		cudaMemcpy(x_GPU, flattenedX, X_rows * X_cols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(weights_GPU, flattenWeights, architecture[0] * architecture[1] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(biases_GPU, array_biases[0], architecture[1] * sizeof(float), cudaMemcpyHostToDevice);

		int blockSize = 8;
		forwardPropagation(blockSize, x_GPU, weights_GPU,biases_GPU, result_GPU, X_rows, X_cols, architecture[1]);

		cudaMemcpy(result, result_GPU, X_rows * architecture[1] * sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "After adding biases" << std::endl;
		for (int i = 0; i < X_rows; i++)
		{
			for (int j = 0; j < architecture[1]; j++)
			{
				std::cout << result[i * architecture[1] + j] << " ";
			}
			std::cout << std::endl;
		}


		// DELETING
		cudaFree(x_GPU);
		cudaFree(weights_GPU);
		cudaFree(result_GPU);
		cudaFree(biases_GPU);

		delete[] result;
		delete[] flattenedX;
		delete[] flattenWeights;

		delete[] Z[0];
		delete[] Z;
	
	#else
		forward_propagation(X, array_weights[0], array_biases[0], Z[0], X_rows, X_cols, architecture[1]);
		if (a_len >= 2) {
			sigmoid(Z[0], X_rows, architecture[1]);
		
			for (int i = 1; i < a_len; i++) {
				Z[i] = new float* [X_rows];
				for (int j = 0; j < X_rows; j++)
					Z[i][j] = new float[architecture[i+1]];
				forward_propagation(Z[i-1], array_weights[i], array_biases[i], Z[i], X_rows, architecture[i], architecture[i+1]);
				//if a hidden layer - do activation function
				if(i!=a_len-1)
					sigmoid(Z[i], X_rows, architecture[i+1]);
			}
		}

		softmax(Z[a_len-1], X_rows, architecture[a_len]);

		// Compute loss

		float cross_loss = cross_entropy_loss(Y, Z[a_len - 1], X_rows, architecture[a_len]);
		std::cout << "EPOCH: " << 1 << '\t' << "Loss: " << cross_loss << std::endl;

		#ifdef DEBUG
			for (int i = 0; i < a_len; i++) {
				std::cout << "Weights: " << i << std::endl;
				for (int j = 0; j < X_rows; j++) {
					for (int k = 0; k < architecture[i + 1]; k++) {
						std::cout << Z[i][j][k] << ' ';
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				std::cout << std::endl;
			}
		#endif // DEBUG
	
		// Back propagation
		back_propagation(X, Y, Z, X_rows, X_cols);

		/////////////////////////////////////////////////////////////////
		//UPDATE WEIGHTS
		update_weights(lr);
		#ifdef DEBUG

			for (int i = 0; i < a_len; i++) {
				std::cout << " Weights: " << i << std::endl;
				for (int j = 0; j < architecture[i]; j++) {
					for (int k = 0; k < architecture[i + 1]; k++) {
						std::cout << array_weights[i][j][k] << ' ';
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				std::cout << std::endl;
			}
			for (int j = 0; j < a_len; j++) {
				std::cout << " Biases: " << j << std::endl;
				for (int k = 0; k < architecture[j + 1]; k++) {
					std::cout << array_biases[j][k] << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			std::cout << std::endl;
		#endif // DEBUG

		if (epochs == 1 || epochs < 1)
			return;

		//First epoch was before
		for (int i = 0; i < epochs-1; i++) {
			//FORWARD PROPAGATION
			std::cout << "EPOCH: " << i+2 << '\t';
			forward_propagation(X, array_weights[0], array_biases[0], Z[0], X_rows, X_cols, architecture[1]);
			if (a_len >= 2) {
				sigmoid(Z[0], X_rows, architecture[1]);
				for (int i = 1; i < a_len; i++) {
					forward_propagation(Z[i - 1], array_weights[i], array_biases[i], Z[i], X_rows, architecture[i], architecture[i + 1]);
					//if a hidden layer - do activation function
					if (i != a_len - 1)
						sigmoid(Z[i], X_rows, architecture[i + 1]);
				}
			}
			softmax(Z[a_len - 1], X_rows, architecture[a_len]);
			// COMPUTE LOSS
			float cross_loss = cross_entropy_loss(Y, Z[a_len - 1], X_rows, architecture[a_len]);
			std::cout << "Loss: " << cross_loss << std::endl;
			//
			// 
			//BACK PROPAGATION
			back_propagation(X, Y, Z, X_rows, X_cols);

			//UPDATE WEIGHTS
			update_weights(lr);
		}

	/////////////////////////////////////////////////////////////////
	// DELETING
	for (int i = 0; i < a_len; i++) {

		for (int j = 0; j < X_rows; j++) {
			delete[] Z[i][j];
		}
		delete[] Z[i];
	}
	delete[] Z;
	/////////////////////////////////////////////////////////////////
	#endif 


}

float** Neural_Network::predict(float** X, int X_rows, int X_cols)
{
	float*** Z = new float** [a_len];
	// init first result XW weights
	Z[0] = new float* [X_rows];
	for (int j = 0; j < X_rows; j++)
		Z[0][j] = new float[architecture[1]];

	forward_propagation(X, array_weights[0], array_biases[0], Z[0], X_rows, X_cols, architecture[1]);
	if (a_len >= 2) {
		sigmoid(Z[0], X_rows, architecture[1]);

		for (int i = 1; i < a_len; i++) {
			Z[i] = new float* [X_rows];
			for (int j = 0; j < X_rows; j++)
				Z[i][j] = new float[architecture[i + 1]];
			forward_propagation(Z[i - 1], array_weights[i], array_biases[i], Z[i], X_rows, architecture[i], architecture[i + 1]);
			//if a hidden layer - do activation function
			if (i != a_len - 1)
				sigmoid(Z[i], X_rows, architecture[i + 1]);
		}
	}
	softmax(Z[a_len - 1], X_rows, architecture[a_len]);


	float** Y_pred = new float* [X_rows];
	//nb_of_classes
	for (int i = 0; i < X_rows; i++) {
		Y_pred[i] = new float[architecture[a_len]];
	}

	for (int i = 0; i < X_rows; i++) 
		for (int j = 0; j < architecture[a_len]; j++) 
			Y_pred[i][j] = Z[a_len - 1][i][j];

	// DELETING
	for (int i = 0; i < a_len; i++) {

		for (int j = 0; j < X_rows; j++) {
			delete[] Z[i][j];
		}
		delete[] Z[i];
	}
	delete[] Z;

	return Y_pred;
}

float Neural_Network::evaluate(float** X, int* Y_true, int X_rows, int X_cols)
{
	float acc = 0.0f;
	float** Y_pred = predict(X, X_rows, X_cols);

	int* Y_pred_num = to_numerical(Y_pred, X_rows, architecture[a_len]);

	acc = accuracy(Y_pred_num, Y_true, X_rows);

	for (int i = 0; i < X_rows; i++)
		delete[] Y_pred[i];
	delete[] Y_pred;
	delete[] Y_pred_num;
	return acc;
}

void Neural_Network::test(int rows,int cols)
{
	float* A = new float[rows * cols];
	float* B = new float[architecture[0] * architecture[1]];

	// ... (populate matrices A and B)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[i * cols + j] = i * cols + j;
			std::cout << A[i * cols + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
	for (int i = 0; i < architecture[0]; i++) {
		for (int j = 0; j < architecture[1]; j++) {
			B[i * architecture[1] + j] = i * architecture[1] + j;
			std::cout << B[i * architecture[1] + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	// Allocate memory for matrices A and B on the device
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, rows * cols * sizeof(float));
	cudaMalloc((void**)&d_B, architecture[0] * architecture[1] * sizeof(float));
	cudaMalloc((void**)&d_C, rows * architecture[1] * sizeof(float));

	// Copy matrices from host to device
	cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, architecture[0] * architecture[1] * sizeof(float), cudaMemcpyHostToDevice);

	// Define grid and block dimensions
	dim3 DimGrid((architecture[1] - 1) / 8 + 1, (rows- 1) / 8 + 1, 1);
	dim3 DimBlock(8, 8, 1);

	int blockSize = 8;

	// Launch the kernel
	//Wrapper::matmul_wrap(numBlocks, blockSize, d_A, d_B, d_C, rows, cols, architecture[1]);
	matrixMultiplication(blockSize, d_A, d_B, d_C, rows, cols, architecture[1]);
	// Copy the result matrix back to the host
	float* C = new float[rows * architecture[1]];
	cudaMemcpy(C, d_C, rows * architecture[1] * sizeof(float), cudaMemcpyDeviceToHost);


	std::cout << "Result Matrix C:" << std::endl;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < architecture[1]; j++) {
			std::cout << C[i * architecture[1] + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
	
	// Print the result matrix C

	// Free host and device memory
	delete[] A;
	delete[] B;
	delete[] C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
