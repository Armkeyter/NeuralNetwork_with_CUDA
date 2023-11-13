# Flow of the programm

[Report link](https://www.overleaf.com/5644215827cjqmysvcmkwr#fcb819)

We need to implement neural network with CUDA.
What do we need to have to start:
Firstly, we will try to do classification task because we already know how to do it.

## Sequential part

1. Find an input test data 		
	- (it can be just points like x=[0,0] and y={0,1}) 
2. Write how to read from file the dataset.
3. Specify how we would like to create a neural network 
	- (it can be just as an array of arrays but it is ugly or we can create a class of a model
which will have layers inside and methods for fitting predicting with a model and much more.
It is done like this in keras)
4. Find a way to randomize weights.
5. Create a function for init a model 	
    - (firstly we can do a model with one hidden layer)
6. Create activation functions and their derivatives
	- Sigmoid, Relu, Tanh, LeakyReLU etc
7. Create a function for learning(back-forward propagation)
	- forward function
	- computing loss
	- backward function
	- update weights
8. Create loss functions
	- mse,mae, crossentropy
9. Create a function for fit and predict.
10. Make a wrapper of time to measure the time it takes to train the model
11. (Optional) Write results of time measure and other info in file for further plots

___Only after everything works sequentially we can do in parallel.___

## Parallel Part

The first that comes in mind is to paralellise:
	
	Fit function
	- matrix multiplication
	- dot product
	- element-wise multiplication
	- bias addition
	Updating weights function

### To add after

We need to add some functions of preprocessing the data (text to categories).

We need to implement support vector machine **SVM**.
Instead of implementing AdaBoost which we mentioned we can implement __Voting ensemble method__


We can think to add also **Convolution layers**.




