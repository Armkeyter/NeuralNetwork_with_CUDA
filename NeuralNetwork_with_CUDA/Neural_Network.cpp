#include "Neural_Network.h"


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
	//Each weight,bias entity
	for (int i = 0; i < architecture_length; i++) {
		//For each row
		for (int j = 0; j < architecture[i]; j++) {
			//For each column
			for (int k = 0; k < architecture[i+1]; k++) {
				array_weights[i][j][k] = k + j + i;
			}
		}
	}
	// init biases

	for (int i = 0; i < architecture_length; i++)
		for(int j=0;j<architecture[i+1];j++)
			array_biases[i][j] = i + j;
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
