#include <iostream>
#include <fstream>
#include "./cuda_kernel.cuh"
#include"./utils.h"
#include"./Neural_Network.h"


int main()
{
    std::string filename = "data_banknote_authentication.txt";
    //std::string filename = "check_read.txt";
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
    int architecture[] = {4,16,2};
    //int architecture[] = { 4,16,32,4,8,16,4,4,16,4,2 };
    // Length of array
    int length = sizeof(architecture) / sizeof(architecture[0])-1;
    std::cout << "Size of the array: " << length << std::endl;
    // Creating NN

    //int features[2] = { 0,1 };
    //MinMaxSacaler(X, rows, cols - 1, features);
    unsigned int seed = 10;
    shuffle(X, Y, rows);
    Neural_Network NN(architecture,length);
    NN.print_weights_biases();
    std::cout << std::endl << std::endl;
    std::cout << "DATA" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            std::cout << X[i][j] <<' ';
        }
        std::cout<<'\t' << Y[i] << std::endl;
    }
    std::cout << std::endl << std::endl;
   
    int** one_hot_Y;
    int num_of_classes = 0;
    float lr = 0.01;
    int epochs = 80;
    one_hot_Y = one_hot_encoding(Y, rows, &num_of_classes);
    float** Y_pred;
    int* Y_pred_num;
    try
    {
        
        NN.fit(X, one_hot_Y, rows, cols - 1, epochs,lr);
        Y_pred = NN.predict(X, rows, cols-1);
        std::cout << "Predicted values" << std::endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < num_of_classes; j++) {
                std::cout << Y_pred[i][j] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

        Y_pred_num = to_numerical(Y_pred, rows, num_of_classes);
        std::cout << "Predicted values numeric" << std::endl;
        for (int i = 0; i < rows; i++) {
            std::cout << Y_pred_num[i] << ' ';
        }
        std::cout << std::endl;

        std::cout << "Accuracy of NN is: " << accuracy(Y, Y_pred_num, rows)<<std::endl;
        float eval = NN.evaluate(X, Y, rows, cols - 1);
        std::cout << "Accuracy of NN (Evaluate function) is: " << eval << std::endl;

        //Deleting dynamic memory
        for (int i = 0; i < rows; ++i) {
            delete[] d_array[i];
            delete[] X[i];
            delete[] Y_pred[i];
            delete[] one_hot_Y[i];
        }
        delete[] d_array;
        delete[] X;
        delete[] Y;
        delete[] Y_pred;
        delete[] one_hot_Y;
        delete[] Y_pred_num;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
