hellomake: NeuralNetwork_with_CUDA.cpp utils.cpp utils.h Neural_Network.cpp Neural_Network.h activations.cpp activations.h
	g++ -o main.exe NeuralNetwork_with_CUDA.cpp utils.cpp Neural_Network.cpp activations.cpp losses.cpp
