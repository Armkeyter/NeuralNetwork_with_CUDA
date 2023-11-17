#include <map>
#include <iostream>
#include <random>


//With every passing day my hatred for c++ grows
//u should take a look at this https://www.youtube.com/watch?v=dQw4w9WgXcQ
float dot_product(float &a, float &b, int length){
    float res;
    for(int i = 0; i < length; i++ ){
        res+= a[i]*b[i];

    }
    return res;
}

float* mat_mul(float &a, float &b, const int rows_a, const int cols_b){
    res = new float[rows_a][cols_b];
    for(i = 0; i < rows_a ; i++){
        for(j = 0; j < cols_b ; j++){

            //TODO :()
            //I don't have time but I think i'll build a get_columns function as in the tp and work with dot products to make the code more readable
        }
    }

}


std::map<std::string, float> init_model(const int d_input,const int d_hidden,const int d_output){
    //Something's going weird with the const but todo està bien mi amigo
    std::map<std::string, float> m;
    std::default_random_engine generator;
    std::uniform_real_distribution<int> distribution(0,1);

    int *W1;
    int *b1;
    int *W2;
    int *b2;

    W1 = new float[d_input][d_hidden];
    b1 = new float[][];
    W2 = new float[d_hidden][d_output];
    b2 = new float[][];

    map<int, string> m = {{
                                'W1',
                                W1,
                            },
                            {
                                'b1',
                                b1,
                            },
                            {
                                'W2'
                                W2,
                            },
                            {
                                'b2'
                                b2,
                            }};


    return m;

//Idk how to make it more general so that we can arbitrarily set the numbers of layers etc if u have any idea how to implement this

}