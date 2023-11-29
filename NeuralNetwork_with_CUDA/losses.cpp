#include "losses.h"
float cross_entropy_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols)
{
    float loss = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < Y_rows; i++) {
        temp = 0.0f;
        for (int j = 0; j < Y_cols; j++) {
            temp += -(Y_true[i][j]*log(Y_pred[i][j]));
        }
        loss += temp / Y_rows;
    }
    return loss;
}

float mse_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols)
{
    float loss = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < Y_rows; i++) {
        temp = 0.0f;
        for (int j = 0; j < Y_cols; j++) {
            temp += pow((Y_true[i][j]-(Y_pred[i][j])),2);
        }
        loss += temp / Y_rows;
    }
    return loss;
}

float mae_loss(int** Y_true, float** Y_pred, int Y_rows, int Y_cols)
{
    float loss = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < Y_rows; i++) {
        temp = 0.0f;
        for (int j = 0; j < Y_cols; j++) {
            temp += abs((Y_true[i][j]-(Y_pred[i][j])));
        }
        loss += temp / Y_rows;
    }
    return loss;
}