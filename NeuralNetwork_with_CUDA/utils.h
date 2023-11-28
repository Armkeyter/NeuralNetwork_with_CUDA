#pragma once
#include <string>

float** read_csv_file(std::string filename, int* rows, int* cols);

/**
*One hot encoder encodes labels to the indexes of its values
*@param Y - input labels
*@param rows - Size of Y array
*@param num_of_classes - function will return number of classes to this variable
* @return 2D one_hot_encoded array.
*/
int** one_hot_encoding(int* Y, int rows,int* num_of_classes);

/**
*MinMaxScaler scales the array between given two givven features
*@param X - 2D input array
*@param rows - size of row of X array
*@param cols - size of row of X array
*@param feature_range - array desired range of transformed data i.e. (0,1)
* @return None.
*/
void MinMaxSacaler(float** X, int rows, int cols, int feature_range[2]);