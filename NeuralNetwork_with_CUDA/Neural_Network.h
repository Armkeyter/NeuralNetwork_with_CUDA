#include <iostream>
#pragma once
class Neural_Network
{
private:
	float*** array_weights;
	float** array_biases;
	float*** dW;
	float** db;
	int architecture_length;
	int* architecture;

	/**
	* Dot product between two vectors
	*
	* @param X: input row of data
	* @param W: weights of the model
	* @param size: size of array
	* @return dot_product X*W.
	*/
	float dot_product(float* X, float* W,int size);

	/**
	* Dot product between two vectors
	*
	* @param W: weights of the model
	* @param row_size: size of row of an array
	* @param col_size: size of col of an array
	* @param index: index of column to return
	* @return float* array column of matrix[index].
	*/
	void get_column(float** W, int row_size, int index, float result[]);

	void minus_matrix(float** X, float** Y, int row_size, int col_size);

	void hadamard(float** X, float** Y, int row_size, int col_size);
	float ** hadamard_return(float** X, float** Y, int row_size, int col_size);

	float** matrix_transpose(float**X, const int row_size,const int col_size);
	/**
	* Matrix multiplication between matrix X and W
	* @param X: Input matrix
	* @param W: weights of the model
	* @param row_size: size of row of X array
	* @param col_size: size of col of X array
	* @param W_col_size: size of col of a W array
	* @param index: index of column to return
	* @return float* array column of matrix[index].
	*/
	void matrix_multiplication(float** X,float** W, float** res, int row_size,int col_size,int W_col_size);

	float** matrix_multiplication_return(float** X, float** W, int row_size, int col_size, int W_col_size);

public:
	/**
	* Initialize neural network.
	*
	* @param architecture 1D array with the size of each layer:
	starting from Input till output layer.
	* @return entity of the class.
	*/
	Neural_Network(int* architecture,int size);

	/**
	Deleteing object.
	* @return None.
	*/
	~Neural_Network();

	/**
	* Initialize weights and biases of NN.
	* @return None.
	*/
	void init_weights();

	/**
	* Debug print weights and biases.
	* @return None.
	*/
	void print_weights_biases();


	/**
	* Forward functoin from back_forward propagation.
	* @param X - input data
	* @param W - weights of the model
	* @param b - biases of the model
	* @param res - result of multiplication size of [W_row_size,col_size]
	* @param row_size: size of row of X array
	* @param col_size: size of col of X array
	* @param W_col_size: size of col of a W array
	* @return None.
	*/
	void forward_propagation(float** X, float** W, float* b, float** res, int row_size, int col_size, int W_col_size);

	/**
	* Forward functoin from back_forward propagation.
	* @param X - input data
	* @param Y - input data (labels)
	* @param X_rows: size of row of X array
	* @param X_cols: size of col of X array
	* @return None.
	*/
	void fit(float** X, float* Y,int X_rows,int X_cols);


	void backpropagation(float* learning_rate, float*** Z,int size_Y, int nb_classes, float** X, int X_rows, int X_cols, float* Y_labels);
	/**
	*Full backpropagation update on the network
	*
	*
	*/

	float compute_loss(float** Y_labels, float ** Y, int* size_Y, int* nb_classes);
	/**
	*Computes the cross entropy
	* note : This loss assumes that we are working with outputs which are probabilities as zeros will make the computation fail
	* note2 : we can switch to MSE if this is causing issues in the early tests
	*@param Y_labels  labels
	*@param Y - Y outputs from the network
	*/
};

