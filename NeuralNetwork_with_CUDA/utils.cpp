#include <iostream>
#include <fstream>
#include "./utils.h"

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
        for (int j = 0; j < numCols; j++) {
            getline(file, line, ',');
            float num_double = std::stof(line);
            data[i][j] = num_double;
            std::cout << data[i][j] << ' ';
        }
        i++;
        std::cout << std::endl;
    }

    file.close();
    *rows = numRows;
    *cols = numCols;
    return data;
}