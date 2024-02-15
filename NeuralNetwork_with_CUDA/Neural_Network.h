#include <iostream>
#pragma once
#include "./utils.h"



class Neural_Network
{
private:
	float*** array_weights;
	float** array_biases;
	float*** dW;
	float** db;
	int a_len;
	int* architecture;



	float** matrix_copy(float** X, int row_size, int col_size);

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
	float** minus_matrix_return(float** X, float** Y, int row_size, int col_size);
	float* minus_vector_return(float* X, float* Y, int row_size);

	void hadamard(float** X, float** Y, int row_size, int col_size);
	float** hadamard_return(float** X, float** Y, int row_size, int col_size);

	float** scalar_multiply_return(float** X, float* y, int row_size, int col_size);
	float* scalar_multiply_return(float* X, float* y, int row_size);
	float** matrix_transpose(float** X, const int row_size, const int col_size);
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

	/**
	* Updates array_weights, array_biases with dW and db.
	* @param lr - learnign rate
	* @return None.
	*/
	void update_weights(float lr);

	/**
	* Back propagatoin that countd dW and db of NeuralNetwork class
	* @param X - input data
	* @param Y - labels
	* @param Z - activation functions
	* @param row_size: size of row of X array
	* @param col_size: size of col of X array
	* @return None.
	*/
	void back_propagation(float** X,int** Y,float*** Z,int rows,int cols);
	/**
	* Back propagatoin in GPU that count dW and db of NeuralNetwork class
	* @param X - input data
	* @param Y - labels
	* @param Z - activation functions
	* @param row_size: size of row of X array
	* @param col_size: size of col of X array
	* @return None.
	*/
	//void back_propagation_GPU(int blockSize, float* x_GPU, float** Z_GPU, float** weights_GPU,
	//	float** biases_GPU, float lr, float* Y, int X_rows, int X_cols);

	void back_propagation_GPU(int blockSize, float* x_GPU, float* Y_GPU, float** Z_GPU,
		float** weights_GPU, float** dW_GPU, float** biases_GPU, float** db_GPU, float** new_weights, float lr, int X_rows, int X_cols);

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

public:
	/**
	* Initialize neural network.
	*
	* @param architecture 1D array with the size of each layer:
	starting from Input till output layer.
	* @return entity of the class.
	*/
	Neural_Network(int* architecture,int size,int seed=NULL);

	/**
	Deleteing object.
	* @return None.
	*/
	~Neural_Network();

	/**
	* Initialize weights and biases of NN.
	* @param seed = random seed, if seed is not given than each time it is random
	* @return None.
	*/
	void init_weights(unsigned int seed=NULL);

	/**
	* Debug print weights and biases.
	* @return None.
	*/
	void print_weights_biases();
	
	/**
	* Forward functoin from back_forward propagation.
	* @param X - input data
	* @param Y - input data (labels)
	* @param X_rows: size of row of X array
	* @param X_cols: size of col of X array
	* @param lr: learnign rate of the algorithm
	* @return None.
	*/
	void fit(float** X, int** Y,int X_rows,int X_cols,int epochs=1, float lr=0.001);


	/**
	* Predict function that computes forward propagation on the data
	* @param X - input data
	* @param X_rows: size of row of X array
	* @param X_cols: size of col of X array
	* @return float.
	*/
	float** predict(float** X, int X_rows, int X_cols);
	
	/**
	* Evaluate function that predict and returns the accuracy of the model
	* @param X - input data
	* @param Y_true - true labels of Y (in numeric encoding)
	* @param X_rows: size of row of X array
	* @param X_cols: size of col of X array
	* @return float.
	*/
	float evaluate(float** X,int* Y_true, int X_rows,int X_cols);

	float compute_loss(float** Y_labels, float ** Y, int* size_Y, int* nb_classes);
	/**
	*Computes the cross entropy
	* note : This loss assumes that we are working with outputs which are probabilities as zeros will make the computation fail
	* note2 : we can switch to MSE if this is causing issues in the early tests
	*@param Y_labels  labels
	*@param Y - Y outputs from the network
	*/
	void test(int rows, int cols);
};

