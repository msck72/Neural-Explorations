#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <thread>
#include <cassert>

#include "inference_tensor.h"
#include "conv.h"
#include "pool_layers.h"

using namespace std;

struct BasicBlock{
    size_t input_channels;
    size_t output_channels;
    size_t stride;
    size_t padding;
    ConvLayer conv1;
    ConvLayer conv2;
    ConvLayer conv3;

    BasicBlock(size_t input_channels, size_t output_channels, size_t stride): conv1(output_channels, 3, input_channels, stride, 1), conv2(output_channels, 3, output_channels, 1, 1), conv3(output_channels, 1, input_channels, stride, 0){
        this->input_channels = input_channels;
        this->output_channels = output_channels;
        this->stride = stride;
        this->padding = padding;
        // conv1 = ConvLayer(output_channels, 3, input_channels, stride, 1);
        // conv2 = ConvLayer(output_channels, 3, output_channels, 1, 1);
        
        // if(input_channels != output_channels){
        //     conv3 = ConvLayer(output_channels, 1, input_channels, stride, 0);
        // }

    }

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        assert(input_tensor.shape[0] == input_channels);

        InferenceTensor out = conv1(input_tensor);
        out = out.relu();
        out = conv2(out);
        
        InferenceTensor identity = input_tensor;
        if(input_channels != output_channels){
            identity = conv3(input_tensor);
        }
        out = out + identity;

        out = out.relu();
        return out;
    }
};


struct Stem{
    ConvLayer conv;
    MaxPoolLayer pool;
    
    Stem(int in_channels, int out_channels, int kernel_size, int conv_stride, int conv_padding, int pool_kernel_size, int pool_stride, int pool_padding): conv(out_channels, kernel_size, in_channels, conv_stride, conv_padding), pool(pool_kernel_size, pool_stride, pool_padding){
        
    }

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor out = conv(input_tensor);
        out = out.relu();
        out = pool(out);
        return out;
    }
};

struct ClassifierHead{
    AdaptiveAvgPoolLayer pool;
    InferenceTensor FC;

    ClassifierHead(size_t in_channels, size_t num_classes): pool(1), FC({in_channels, num_classes}){}

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor out = pool(input_tensor);
        out = out.flatten();
        out = out.matmul(FC);
        return out;
    }
};