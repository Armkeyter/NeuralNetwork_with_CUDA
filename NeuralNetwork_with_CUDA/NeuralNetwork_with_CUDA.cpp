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
    float* Y = new float[rows];
    float** X = new float*[rows];
    for (int i = 0; i < rows; i++) {
        Y[i] = d_array[i][cols-1];
        X[i] = new float[cols];
        for (int j = 0; j < cols-1; j++)
            X[i][j] = d_array[i][j];
    }

    // Creating architecture
    int architecture[] = {4,4,8,2};
    //4,4,16
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
    
    //Deleting dynamic memory
    for (int i = 0; i < rows; ++i) {
        delete[] d_array[i];
        delete[] X[i];
    }
    delete[] d_array;
    delete[] X;
    delete[] Y;
    return 0;
}
