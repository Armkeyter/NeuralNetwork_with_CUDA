#include "Neural_Network.h"
#include "./activations.h"
#include <random>

float **Neural_Network::matrix_copy(float** X, int row_size, int col_size){
	float ** res = new float*[row_size];
	for(int i = 0; i<row_size;i++){
		res[i] = new float[col_size];
		for(int j = 0; j<col_size; j++){
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



void Neural_Network::get_column(float** W, int row_size, int index,float result[])
{
	for (int i = 0; i < row_size; ++i)
		result[i] = W[i][index];
}

void Neural_Network::minus_matrix(float** X, float** Y, int row_size, int col_size){
	for(int i = 0; i < row_size;i++){
		for(int j = 0; j < col_size;j++){
			X[i][j] = X[i][j] - Y[i][j];
		}
	}
}

float** Neural_Network::minus_matrix_return(float** X, float** Y, int row_size, int col_size){
	for(int i = 0; i < row_size;i++){
		for(int j = 0; j < col_size;j++){
			X[i][j] = X[i][j] - Y[i][j];
		}
	}
	return X;
}

void Neural_Network::hadamard(float** X, float** Y, int row_size, int col_size){
	for(int i = 0; i < row_size;i++){
		for(int j = 0; j < col_size;j++){
			X[i][j] = X[i][j] * Y[i][j];
		}
	}
}


float** Neural_Network::hadamard_return(float** X, float** Y, int row_size, int col_size){
	for(int i = 0; i < row_size;i++){
		for(int j = 0; j < col_size;j++){
			X[i][j] = X[i][j] * Y[i][j];
		}
	}
	return X;
}


float** Neural_Network::matrix_transpose(float**X, int row_size, int col_size){
	float** res = new float*[col_size];
	for (int i = 0; i < col_size; i++){
		res[i] = new float[row_size];
	}
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			res[j][i] = X[i][j];
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
{	float **res = new float*[row_size];
	for (int i = 0; i<row_size; i++){
		res[i] = new float[W_col_size];
	}
	float* column = new float[col_size];
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < W_col_size; j++) {
			get_column(W, col_size, j, column);
			res[i][j] = dot_product(X[i],column,col_size);
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
	
	architecture_length = size;
	this->architecture = new int[architecture_length+1];
	array_weights = new float** [architecture_length];
	array_biases = new float* [architecture_length];
	dW = new float** [architecture_length];
	db = new float* [architecture_length];
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

void Neural_Network::backpropagation(float* learning_rate, float*** Z,int size_Y, int nb_classes, float** X, int X_rows, int X_cols, float* Y_labels){


	// I got tired of typing it.
	int l = architecture_length;

	//definition of delta_i, i'm pretty sure it is a very wrong way to do it and i'm thinking of making it an attribute of the class
	float** delta_i = new float*[size_Y];

	for (int i = 0; i<size_Y;i++){
		delta_i[i] = new float[nb_classes];
	}

	//I need to keep track of the size of delta
	int current_size_row;
	int current_size_col;
	current_size_row = size_Y;
	current_size_col = nb_classes;


	//Initialisation, of the first delta
	delta_i = minus_matrix_return(Z[l],Z[architecture_length],size_Y,nb_classes);
	for(int i =0; i< l; i++){

		//I make a copy of delta_i so that i can delete and recreate it , I am sorry this is very ugly
		float** delta_previous = matrix_copy(delta_i,current_size_row,current_size_col);
		
		for (int i = 0; i<size_Y;i++){
			delete[] delta_i[i];
		}
		delete[] delta_i;
		

		//I remake a delta_i with the right sizes to prepare for further computation
		float** delta_i = new float*[X_rows];
		for (int i = 0; i<size_Y;i++){
			delta_i[i] = new float[architecture[l-i-1]];
		}

		//We compute the delta corresponding to the current step using the delta of the previous step

		delta_i = hadamard_return(sigmoid_return(Z[l-i-1],X_rows,architecture[l-i-1], true),matrix_multiplication_return(delta_i,matrix_transpose(array_weights[l-i-1],X_rows,architecture[l-i-1]),X_rows,X_cols,architecture[l-i-1]),X_rows,architecture[l-i-1]);

		//I update my delta sizes
		current_size_row = X_rows;
		current_size_col = architecture[l-i-1];


		dW[l-i-1] = matrix_multiplication_return(matrix_transpose(X,X_rows,X_cols),delta_i,X_cols,X_rows,architecture[l-i-1]);


		//Sum along axis 0 on delta_i to get db
		float* res = new float[current_size_col];
		for(int j = 0; j<current_size_col;j++){
			for(int i = 0; i< current_size_row;i++){
		
				res[j] = res[j] + delta_i[i][j];
			}
		}


		db[l-i-1] = res;




		//I redelete the delta copy since it is now useless (sorry again)
		for (int i = 0; i<X_rows;i++){
			delete[] delta_previous[i];
		}
		delete[] delta_previous;
		
		
		//TODO : update W and b w/ regard to dW and db in the class attributes


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
