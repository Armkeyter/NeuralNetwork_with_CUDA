#include <iostream>
#pragma once
class Neural_Network
{
private:
	float*** array_weights;
	float** array_biases;
	int architecture_length;
	int* architecture;
public:
	/**
	* Initialize neural network.
	*
	* @param architecture 1D array with the size of each layer:
	starting from Input till output layer.
	* @return entity of the clall.
	*/
	Neural_Network(int* architecture,int size);

	/**
	Deleteing object.
	* @return None.
	*/
	~Neural_Network();

	float*** get_weights() {
		return array_weights;
	}

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
};

