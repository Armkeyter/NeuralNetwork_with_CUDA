#include <iostream>
#include <fstream>
#include "./cuda_kernel.cuh"
#include"./utils.h"
#include"./Neural_Network.h"

int main()
{
    //std::string filename = "data_banknote_authentication.txt";
    std::string filename = "check_read.txt";
    int rows, cols;

    // Read data from the file
    float** d_array = read_csv_file(filename, &rows, &cols);
    std::cout << "Rows: " << rows << " Cols: " << cols<< std::endl << std::endl;
    
    // Split d_array into X,Y data
    int* Y = new int[rows];
    float** X = new float*[rows];
    for (int i = 0; i < rows; i++) {
        Y[i] = (int)d_array[i][cols-1];
        X[i] = new float[cols];
        for (int j = 0; j < cols-1; j++)
            X[i][j] = d_array[i][j];
    }

    // Creating architecture
    int architecture[] = {4,4,8,2};
    //int architecture[] = { 4,16,32,4,8,16,4,4,16,4,2 };
    // Length of array
    int length = sizeof(architecture) / sizeof(architecture[0])-1;
    std::cout << "Size of the array: " << length << std::endl;
    // Creating NN
    Neural_Network NN(architecture,length);
    NN.print_weights_biases();
    std::cout << std::endl << std::endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            std::cout << X[i][j] <<' ';
        }
        std::cout<<'\t' << Y[i] << std::endl;
    }
    try
    {
        NN.fit(X, Y, rows, cols - 1);

    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    float** one_hot_Y;
    int num_of_classes = 0;
    one_hot_Y = one_hot_encoding(Y, rows,&num_of_classes);
    std::cout<<"One hot encoding: " << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < num_of_classes; j++) {
            std::cout << one_hot_Y[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    int features[2] = { 0,1 };
    MinMaxSacaler(X,rows,cols-1,features);
    std::cout << "MinMaxScaler: " << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols-1; j++) {
            std::cout << X[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    //Deleting dynamic memory
    for (int i = 0; i < rows; ++i) {
        delete[] d_array[i];
        delete[] X[i];
        delete[] one_hot_Y[i];
    }
    delete[] d_array;
    delete[] X;
    delete[] Y;
    delete[] one_hot_Y;
    return 0;
}
