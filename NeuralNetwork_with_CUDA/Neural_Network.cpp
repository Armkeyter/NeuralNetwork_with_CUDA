#include "Neural_Network.h"
#include "./activations.h"
#include <random>


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

Neural_Network::Neural_Network(int* architecture,int size)
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
	init_weights();
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

void Neural_Network::fit(float** X, int** Y,int X_rows,int X_cols)
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

	for (int i = 0; i < a_len; i++) {
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
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	//softmax - Y
	for (int i = 0; i < X_rows; i++) 
		for (int j = 0; j < architecture[a_len]; j++) 
			Z[a_len-1][i][j] -= Y[i][j];

	std::cout << "Softmax-Y "<< std::endl;
	for (int i = 0; i < X_rows; i++) {
		for (int j = 0; j < architecture[a_len]; j++) {
			std::cout << Z[a_len - 1][i][j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;


	float sum = 0.0f;
	//transposed sigmoid activation
	float** delta_T = matrix_copy(Z[a_len - 1], X_rows, X_cols);
	for (int i = 0; i < a_len-1; i++) {
		float** a_prev = matrix_transpose(Z[a_len - i- 2], X_rows, architecture[a_len-i - 1]);
		
		//DW
		matrix_multiplication(a_prev, delta_T, dW[i], architecture[a_len-i -1],
			X_rows, architecture[a_len-i]);

		for (int j = 0; j < architecture[a_len-i - 1]; j++) {
			delete[] a_prev[j];
		}
		delete[] a_prev;

		// DW/N
		for (int j = 0; j < architecture[a_len - i - 1]; j++)
			for (int k = 0; k < architecture[a_len-i]; k++)
				dW[i][j][k] /= X_rows;
		//DB
		for (int j = 0; j < architecture[a_len - i]; j++) {
			sum = 0.0f;
			for (int k = 0; k <X_rows; k++) {
				sum += delta_T[k][j];
			}
			db[i][j] = sum / X_rows;
		}

		std::cout << "DW " << i << ':' << std::endl;
		for (int j = 0; j < architecture[a_len - i - 1]; j++) {
			for (int k = 0; k < architecture[a_len-i]; k++) {
				std::cout << dW[i][j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "Db " << i << ':' << std::endl;
		for (int j = 0; j < architecture[a_len-i]; j++)
			std::cout << db[i][j] << ' ';
		std::cout << std::endl;
		std::cout << std::endl;


		//DZ
		sigmoid(Z[a_len-i - 2], X_rows, architecture[a_len-i - 1], true);
		// W.T
		float** w_t = matrix_transpose(array_weights[a_len-i - 1], architecture[a_len-i-1], architecture[a_len-i]);

		// np.dot(softmax,weights[a_len-1].T)
		float** temp = new float* [X_rows];
		for (int j = 0; j < X_rows; j++) {
			temp[j] = new float[architecture[a_len - i - 1]];
		}
		matrix_multiplication(delta_T, w_t, temp, X_rows, architecture[a_len-i], architecture[a_len - i - 1]);
		// np.multiply(np.dot(delta_T, w2.T), der_sigmoid(sig))
		hadamard(temp, Z[a_len-i - 2], X_rows, architecture[a_len-i - 1]);

		for (int j = 0; j < X_rows; j++) {
			delete[] delta_T[j];
		}
		for (int j = 0; j < X_rows; j++) {
			delta_T[j] = new float[architecture[a_len-i - 1]];
			for (int k = 0; k < architecture[a_len - i - 1]; k++) {
				delta_T[j][k] = temp[j][k];
			}
		}
		std::cout << "dZ "<<i<<':' << std::endl;
		for (int j = 0; j < X_rows; j++) {
			for (int k = 0; k < architecture[a_len - i - 1]; k++) {
				std::cout << delta_T[j][k] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;
		for (int j = 0; j < architecture[a_len - i]; j++) {
			delete[] w_t[j];
		}
		delete[] w_t;
		for (int j = 0; j < X_rows; j++) {
			delete[] temp[j];
		}
		delete[] temp;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	//LAST CASE
	
	//Counting W last
	float** X_T = matrix_transpose(X, X_rows, X_cols);
	matrix_multiplication(X_T, delta_T, dW[a_len-1], X_cols, X_rows, architecture[1]);

	//// dW / N - num of examples
	for (int i = 0; i < X_cols; i++)
		for (int j = 0; j < architecture[1]; j++)
			dW[a_len - 1][i][j] /= X_rows;

	//Count db last
	for (int i = 0; i < architecture[1]; i++) {
		sum = 0.0f;
		for (int j = 0; j < X_rows; j++) {
			sum += delta_T[j][i];
		}
		db[a_len - 1][i] = sum / X_rows;
	}

	std::cout << "DW_Last: " << std::endl;
	for (int j = 0; j < X_cols; j++) {
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
	

	/////////////////////////////////////////////////////////////////
	//UPDATE WEIGHTS



	/////////////////////////////////////////////////////////////////
	// DELETING
	for (int i = 0; i < X_rows; i++) {
		delete[] delta_T[i];
	}
	delete[] delta_T;

	for (int i = 0; i < X_cols; i++) {
		delete[] X_T[i];
	}
	delete[] X_T;

	for (int i = 0; i < a_len; i++) {

		for (int j = 0; j < X_rows; j++) {
			delete[] Z[i][j];
		}
		delete[] Z[i];
	}
	delete[] Z;
	//////////////////////////////////////////////////

}

void Neural_Network::backpropagation(float* learning_rate, float*** Z, int size_Y, int nb_classes, float** X, int X_rows, int X_cols, float* Y_labels) {
	// I got tired of typing it.
	int l = a_len;
	//definition of delta_i, i'm pretty sure it is a very wrong way to do it and i'm thinking of making it an attribute of the class
	float** delta_i = new float* [size_Y];

	for (int i = 0; i < size_Y; i++) {
		delta_i[i] = new float[nb_classes];
	}	
	//I need to keep track of the size of delta
	float current_size_row;
	float current_size_col;
	current_size_row = size_Y;
	current_size_col = nb_classes;


	//Initialisation, of the first delta
	delta_i = minus_matrix_return(Z[l], Z[a_len], size_Y, nb_classes);
	for (int i = 0; i < l; i++) {

		//I make a copy of delta_i so that i can delete and recreate it , I am sorry this is very ugly
		float** delta_previous = matrix_copy(delta_i, current_size_row, current_size_col);

		for (int i = 0; i < size_Y; i++) {
			delete[] delta_i[i];
		}
		delete[] delta_i;


		//I remake a delta_i with the right sizes to prepare for further computation
		float** delta_i = new float* [X_rows];
		for (int i = 0; i < size_Y; i++) {
			delta_i[i] = new float[architecture[l - i - 1]];
		}

		//We compute the delta corresponding to the current step using the delta of the previous step
		delta_i = hadamard_return(sigmoid_return(Z[l - i - 1], X_rows, architecture[l - i - 1], true), matrix_multiplication_return(delta_i, matrix_transpose(array_weights[l - i - 1], X_rows, architecture[l - i - 1]), X_rows, X_cols, architecture[l - i - 1]), X_rows, architecture[l - i - 1]);

		//I update my delta sizes
		current_size_row = X_rows;
		current_size_col = architecture[l - i - 1];


		dW[l - i - 1] = matrix_multiplication_return(matrix_transpose(X, X_rows, X_cols), delta_i, X_cols, X_rows, architecture[l - i - 1]);

		//Sum along axis 0 on delta_i to get db
		float* res = new float[current_size_col];
		for (int j = 0; j < current_size_col; j++) {
			for (int i = 0; i < current_size_row; i++) {

				res[j] = res[j] + delta_i[i][j];
			}
		}


		db[l - i - 1] = res;


		//I redelete the delta copy since it is now obsolete (sorry again)
		for (int i = 0; i < X_rows; i++) {
			delete[] delta_previous[i];
		}
		delete[] delta_previous;


		//TODO : update W and b w/ regard to dW and db in the class attributes
		array_weights[l - i - 1] = minus_matrix_return(array_weights[l - i - 1], scalar_multiply_return(dW[l - i - 1], learning_rate, X_cols, architecture[l - i - 1]), X_cols, architecture[l - i - 1]);
		array_biases[l - i - 1] = minus_vector_return(array_biases[l - i - 1], scalar_multiply_return(db[l - i - 1], learning_rate, current_size_col), current_size_col);

	}
	delete[] delta_i;

}

float Neural_Network::compute_loss(float** Y_labels, float ** Y, int* size_Y, int* nb_classes){
// THIS ASSUMES OUR Y IS IN ONE HOT ENCODING 
	float res = 0;

	for (int i = 0; i < *size_Y; i++){
		for (int j = 0; j < *nb_classes; j++){
			res = res + Y_labels[i][j]*(Y[i][j]);
		}
	}
	return res;
}
