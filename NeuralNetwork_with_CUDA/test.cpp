#include <map>
#include <iostream>
#include <random>

std::map<std::string, float> init_model(const int d_input,const int d_hidden,const int d_output){

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



}