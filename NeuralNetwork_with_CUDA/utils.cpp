#include <iostream>
#include <fstream>
#include <cfloat>
#include "./utils.h"


void write_into_file(std::string filename, float res){
    std::ofstream file;
    file.open(filename, std::ios::app);
    if (!file.is_open()) {
        std::cout << "Doesn't find a file" << std::endl;
    } 
    std::cout << "File has been opened" << std::endl;
    file << res <<";"<< std::endl;
    file.close();
}

float** read_csv_file(std::string filename, int* rows, int* cols) {
    std::ifstream file;
    file.open(filename);
    if (!file.is_open()) {
        std::cout << "Doesn't find a file" << std::endl;
        return NULL;
    }
    std::cout << "File has been opened" << std::endl;
    int numRows = 0;// num of ','
    int numCols = 0; // num of cols
    std::string line;
    if (std::getline(file, line)) {
        for (char ch : line) {
            if (ch == ',') {
                ++numCols;
            }
        }
        ++numCols;// if there are 2 ',' then it divides 3 values so increment  
        ++numRows;
    }
    if (numRows == 0) {
        // Row is empty or without any ','
        return NULL;
    }
    while (std::getline(file, line)) {
        ++numRows;
    }
    file.close();
    std::cout << "Num of rows: " << numRows << "Num of cols: " << numCols << std::endl;
    if (numCols == 0) {
        // If file is empty raise an error
    }
    ////Create 2D array
    float** data = new float* [numRows];
    for (int i = 0; i < numRows; ++i) {
        data[i] = new float[numCols];
    }
    //// Initialize the elements of the 2D array
    file.open(filename);
    int i = 0;
    while (!file.eof()) {
        for (int j = 0; j < numCols-1; j++) {
            getline(file, line, ',');
            float num_double = std::stof(line);
            data[i][j] = num_double;
        }
        getline(file, line, '\n');
        float num_double = std::stof(line);
        data[i][numCols-1] = num_double;
        i++;
    }

    file.close();
    *rows = numRows;
    *cols = numCols;
    return data;
}

int** one_hot_encoding(int* Y, int rows,int* classes_num)
{
    int min=0, max=0;
    // Counting min label and max to see the span of classes
    for (int i = 0; i < rows; i++) {
        if (Y[i] < min)
            min = Y[i];
        else if (Y[i] > max)
            max = Y[i];
    }
    int num_of_classes = (max - min)+1;
    if (num_of_classes ==  0) {
        throw std::invalid_argument("Array Y contains just one class");
    }
    *classes_num = num_of_classes;
    int** result = new int* [rows];
    for (int i = 0; i < rows; i++) {
        result[i] = new int[num_of_classes];

        for (int j = 0; j < num_of_classes; j++)
            //if value of Y[i] equals to j than we write 1(as represents the class) otherwise 0
            result[i][j] = (Y[i] == j) ? 1 : 0;
    }
    return result;
}

int* to_numerical(float** Y, int Y_rows, int Y_cols)
{
    int* Y_num = new int[Y_rows];
    float max;
    int max_i = 0;
    for (int i = 0; i < Y_rows; i++) {
        max = Y[i][0];
        max_i = 0;
        for (int j = 0; j < Y_cols; j++) {
            if (max < Y[i][j]) {
                max = Y[i][j];
                max_i = j;
            }
        }
        Y_num[i] = max_i;
    }
    return Y_num;
}

float accuracy(int* Y_true, int* Y_pred, int Y_rows)
{
    float accuracy = 0;
    for (int i = 0; i < Y_rows; i++) {
        if (Y_pred[i] == Y_true[i])
            accuracy += 1;
    }
    return accuracy/Y_rows;
}

void MinMaxSacaler(float** X,int rows,int cols, int feature_range[2])
{
    if (feature_range[0] > feature_range[1]) {
        throw std::invalid_argument("First feature is bigger than second");

    }
    
    float X_min = FLT_MAX, X_max= FLT_MIN;
    float* X_min_array = new float[cols];
    float* X_max_array = new float[cols];
    //Find min and max of each column
    for(int i=0;i<cols;i++){
        for (int j = 0; j < rows; j++) {

            if (X[j][i] < X_min) {
                X_min = X[j][i];
            }
            if (X[j][i] > X_max) {
                X_max = X[j][i];
            }
        }
        X_min_array[i] = X_min;
        X_max_array[i] = X_max;
        X_min = FLT_MAX, X_max = FLT_MIN;
    }
    //MinMax Scale formula
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            X[i][j] = (X[i][j] - X_min_array[j]) / (X_max_array[j] - X_min_array[j]);
            X[i][j] *= (feature_range[1] - feature_range[0]) + feature_range[0];
        }
    }

    delete[] X_min_array;
    delete[] X_max_array;
}
