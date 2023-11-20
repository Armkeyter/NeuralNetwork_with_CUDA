#include <iostream>
#include <fstream>
#include "./cuda_kernel.cuh"
#include"./utils.h"
#include"./Neural_Network.h"

int main()
{
    //std::string filename = "data_banknote_authentication.txt";
    //int rows, cols;
    //// Read data from the file
    //float** d_array = read_csv_file(filename, &rows, &cols);
    //std::cout << "Rows: " << rows << " Cols: " << cols<< std::endl;
    
    // Creating architecture
    int architecture[] = {2,4,8};
    // Length of array
    int length = sizeof(architecture) / sizeof(architecture[0])-1;
    std::cout << "Size of the array: " << length << std::endl;
    Neural_Network NN(architecture,length);
    NN.print_weights_biases();

    // Deleting 2D array
    //for (int i = 0; i < rows; ++i) {
    //    delete[] d_array[i];
    //}
    //delete[] d_array;
    return 0;
}
