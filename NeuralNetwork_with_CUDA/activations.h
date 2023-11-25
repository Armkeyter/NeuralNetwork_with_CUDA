#pragma once
/**
* Sigmoid activation function
* @param X - input data
* @param rows: size of row of X array
* @param cols: size of col of X array
* @param is_derivative: if to compute derivative or not (if it is true we assume that array X - is sigmoid output)
* @return None.
*/
float** sigmoid_return(float** X,int rows,int cols,bool is_derivative = false);
void sigmoid(float** X,int rows,int cols,bool is_derivative = false);

/**
* Tanh activation function
* @param X - input data
* @param rows: size of row of X array
* @param cols: size of col of X array
* @param is_derivative: if to compute derivative or not (if it is true we assume that array X - is tanh output)
* @return None.
*/
void tanh(float** X, int rows, int cols,bool is_derivative = false);

/**
* Relu activation function
* @param X - input data
* @param rows: size of row of X array
* @param cols: size of col of X array
* @param is_derivative: if to compute derivative or not
* @return None.
*/
void relu(float** X, int rows, int cols, bool is_derivative = false);

/**
* LeakyRelu activation function
* @param X - input data
* @param alpha - angle on which to multiply when X < 0. Usually between (0,1)
* @param rows: size of row of X array
* @param cols: size of col of X array
* @param is_derivative: if to compute derivative or not
* @return None.
*/
void leakyrelu(float** X, float alpha, int rows, int cols, bool is_derivative = false);

/**
* Softmax activation function
* @param X - input data
* @param rows: size of row of X array
* @param cols: size of col of X array
* @param res: return result of softmax
* @param is_derivative: if to compute derivative or not (if it is true we assume that array X - is sigmoid output)
* @return None.
*/
void softmax(float** X, int rows, int cols,float** res, bool is_derivative = false);

