#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <thread>

#include "inference_tensor.h"

using namespace std;

struct MaxPoolLayer{
    size_t kernal_size;
    size_t stride;
    size_t padding;

    MaxPoolLayer(int kernal_size, int stride = 1, int padding = 0)
        : kernal_size(kernal_size), stride(stride), padding(padding) {}

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor padded_input = input_tensor;
        if(padding > 0){
            size_t input_depth = input_tensor.shape[0];
            size_t input_x = input_tensor.shape[1];
            size_t input_y = input_tensor.shape[2];
            padded_input = InferenceTensor({input_depth, input_x + 2 * padding, input_y + 2 * padding}, 0);
            for(size_t d = 0; d < input_depth; d++){
                for(size_t i = 0; i < input_x; i++){
                    for(size_t j = 0; j < input_y; j++){
                        padded_input.set_item({d, i + padding, j + padding}, input_tensor.get_item({d, i, j}));
                    }
                }
            }
        }
        size_t input_depth = padded_input.shape[0];
        size_t input_x = padded_input.shape[1];
        size_t input_y = padded_input.shape[2];

        InferenceTensor output_tensor({input_depth, (input_x - kernal_size) / stride + 1, (input_y - kernal_size) / stride + 1}, numeric_limits<float>::infinity());

        auto _apply_pool = [&](size_t r, size_t c){
            for(size_t d = 0; d < input_depth; d++){
                float max_value = -numeric_limits<float>::infinity();
                for(size_t i = r; i < r + kernal_size; i++){
                    for(size_t j = c; j < c + kernal_size; j++){
                        max_value = max(max_value, padded_input.get_item({d, i, j}));
                    }
                }
                output_tensor.set_item({d, r / stride, c / stride}, max_value);
            }
        };

        for(size_t row = 0; row <= input_x - kernal_size; row += stride){
            vector<thread> threads;
            for(size_t col = 0; col <= input_y - kernal_size; col += stride){
                threads.emplace_back(_apply_pool, row, col);
            }
            for(auto& t : threads){
                t.join();
            }
        }
        return output_tensor;
    }
};


struct AdaptiveAvgPoolLayer{
    size_t output_size;

    AdaptiveAvgPoolLayer(int output_size)
        : output_size(output_size) {}

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        size_t input_depth = input_tensor.shape[0];
        size_t input_x = input_tensor.shape[1];
        size_t input_y = input_tensor.shape[2];

        InferenceTensor output_tensor({input_depth, output_size, output_size});

        auto _apply_pool = [&](size_t d){
            for(size_t i = 0; i < output_size; i++){
                for(size_t j = 0; j < output_size; j++){
                    float sum_value = 0;
                    for(size_t x = i * input_x / output_size; x < (i + 1) * input_x / output_size; x++){
                        for(size_t y = j * input_y / output_size; y < (j + 1) * input_y / output_size; y++){
                            sum_value += input_tensor.get_item({d, x, y});
                        }
                    }
                    float avg_value = sum_value / ((input_x / output_size) * (input_y / output_size));
                    output_tensor.set_item({d, i, j}, avg_value);
                }
            }
        };

        vector<thread> threads;
        for(size_t d = 0; d < input_depth; d++){
            threads.emplace_back(_apply_pool, d);
        }
        for(auto& t : threads){
            t.join();
        }

        return output_tensor;
    }
};
