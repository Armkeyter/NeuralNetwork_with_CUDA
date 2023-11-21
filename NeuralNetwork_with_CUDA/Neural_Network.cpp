#include "Neural_Network.h"
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
	std::random_device rd;
	std::mt19937 gen(rd());
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

float** Neural_Network::forward_propagation(float** X, float** W, float* b)
{
	
	return nullptr;
}

void Neural_Network::backpropagation(float** X, float*** W, float** b){

}

void Neural_Network::backpropagation_i(float** X, float** W, float* b){

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
