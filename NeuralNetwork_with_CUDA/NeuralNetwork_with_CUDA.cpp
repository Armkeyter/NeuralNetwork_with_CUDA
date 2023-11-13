#include <iostream>
#include <fstream>
#include "./cuda_kernel.cuh"
#include"./utils.h"


int main()
{
    std::string filename = "data_banknote_authentication.txt";
    int rows, cols;
    int* rows_p =&rows;
    int* cols_p = &cols;
    float** d_array = read_csv_file(filename, rows_p, cols_p);
    std::cout << "Rows: " << *rows_p << " Cols: " << *cols_p <<' ' <<rows << std::endl;
    // Deleting 2D array
    for (int i = 0; i < rows; ++i) {
        delete[] d_array[i];
    }
    delete[] d_array;
    return 0;
}
