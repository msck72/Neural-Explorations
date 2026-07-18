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

struct BN_layer{
    InferenceTensor weight;
    InferenceTensor bias;
    InferenceTensor running_mean;
    InferenceTensor running_var;

    BN_layer(size_t channels)
        : weight({channels}, 1.0), bias({channels}, 0.0),
          running_mean({channels}, 0.0), running_var({channels}, 1.0) {
        
        }
        
    InferenceTensor operator()(InferenceTensor input_tensor){
        input_tensor.sub_channelwise_inplace(running_mean);
        InferenceTensor curr_running_var = running_var;
        curr_running_var.add_channelwise_inplace(InferenceTensor(curr_running_var.shape, 1e-5));
        curr_running_var.sqrt_inplace();
        input_tensor.div_channelwise_inplace(curr_running_var);
        input_tensor.mul_channelwise_inplace(weight);
        input_tensor.add_channelwise_inplace(bias);
        return input_tensor;
    }
};

struct BasicBlock{
    size_t input_channels;
    size_t output_channels;
    size_t stride;
    size_t padding;
    ConvLayer conv1;
    ConvLayer conv2;
    ConvLayer conv3;
    vector<BN_layer> bn_layer;

    BasicBlock(
        size_t input_channels,
        size_t output_channels,
        size_t stride
    )
        : conv1(output_channels, 3, input_channels, stride, 1),
        conv2(output_channels, 3, output_channels, 1, 1),
        conv3(output_channels, 1, input_channels, stride, 0)
    {
        this->input_channels = input_channels;
        this->output_channels = output_channels;
        this->stride = stride;
        this->padding = padding;
        
        bn_layer.clear();
        bn_layer.emplace_back(output_channels);
        bn_layer.emplace_back(output_channels);
        if (input_channels != output_channels) {
            bn_layer.emplace_back(output_channels);
        }

    }

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        assert(input_tensor.shape[0] == input_channels);

        InferenceTensor out = conv1(input_tensor);
        out = bn_layer[0](out);
        out = out.relu();
        out = conv2(out);
        out = bn_layer[1](out);

        InferenceTensor identity = input_tensor;
        if(input_channels != output_channels){
            identity = conv3(input_tensor);
            identity = bn_layer[2](identity);
        }
        out = out + identity;

        out = out.relu();
        return out;
    }


    InferenceTensor operator()(const InferenceTensor& input_tensor, map<string, InferenceTensor>& hooks, string block){
        // For debugging purposes, something like pytorch hooks...
        assert(input_tensor.shape[0] == input_channels);

        InferenceTensor out = conv1(input_tensor);
        hooks.emplace(block + "_conv1", out);
        cout << "basic block, conv1 completed, trying bn_layer[0]...\n" << flush;
        out = bn_layer[0](out);
        hooks.emplace(block + "_bn1", out);
        cout << "basic block, bn_layer[0] completed, trying relu...\n" << flush;
        out = out.relu();
        hooks.emplace(block + "_relu", out);
        cout << "basic block, relu completed, trying conv2, out.shape = " << out.shape[0] << " " << out.shape[1] << " " << out.shape[2] << "\n" << flush;
        out = conv2(out);
        hooks.emplace(block + "_conv2", out);
        out = bn_layer[1](out);
        hooks.emplace(block + "_bn2", out);
        cout << "basic block, conv2 completed, trying bn_layer[1]...\n" << flush;
        
        InferenceTensor identity = input_tensor;
        if(input_channels != output_channels){
            identity = conv3(input_tensor);
            identity = bn_layer[2](identity);
        }
        out = out + identity;
        hooks.emplace(block + "_identity", out);

        out = out.relu();
        hooks.emplace(block + "_relu_2", out);
        return out;
    }
};


struct Stem{
    ConvLayer conv;
    BN_layer bn_layer;
    MaxPoolLayer pool;
    
    Stem(int in_channels, int out_channels, int kernel_size, int conv_stride, int conv_padding, int pool_kernel_size, int pool_stride, int pool_padding): conv(out_channels, kernel_size, in_channels, conv_stride, conv_padding), pool(pool_kernel_size, pool_stride, pool_padding), bn_layer(out_channels) {}

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor out = conv(input_tensor);
        out = bn_layer(out);
        out = out.relu();
        out = pool(out);
        return out;
    }
};

struct ClassifierHead{
    AdaptiveAvgPoolLayer pool;
    InferenceTensor FC_weight;
    InferenceTensor FC_bias;

    ClassifierHead(size_t in_channels, size_t num_classes): pool(1), FC_weight({num_classes, in_channels}), FC_bias({num_classes}){}

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor out = pool(input_tensor);
        out = out.flatten();
        out.reshape({out.shape[0], 1});
        out = FC_weight.matmul(out);
        out.reshape({out.shape[0]});
        out = out + FC_bias;
        return out;
    }

    InferenceTensor operator()(const InferenceTensor& input_tensor, map<string, InferenceTensor>& hooks){
        // For debugging purposes, something like pytorch hooks...
        InferenceTensor out = pool(input_tensor);
        hooks.emplace("avgpool", out);
        cout << "ClassifierHead: Pooling done. Shape: " << out.shape[0] << " " << out.shape[1] << " " << out.shape[2] << "\n" << flush;
        out = out.flatten();
        cout << "Flattem done, shape = " << out.shape.size() << "\n";
        out.reshape({out.shape[0], 1});
        cout << "reshape done, shape = " << out.shape[0] << ", 1" << "\n"; 
        // cout << "ClassifierHead: Flattening done. Shape: " << out.shape[0] << " " << out.shape[1] << ", FC.shape: " << FC.shape[0] << " " << FC.shape[1] << "\n" << flush;
        out = FC_weight.matmul(out);
        cout << "matmul done, shape = " << out.shape[0] << ", " << out.shape[1] << "\n";
        out.reshape({out.shape[0]});
        cout << "reshapeing done again\n";
        try{
            cout << "Trying adding bias \n";
            out = out + FC_bias;
            cout << "Bias added \n";
        }
        catch (int e) {
            cout << "Exception Caught: " << e;
        }
        // cout << "ClassifierHead: Matmul done. Shape: " << out.shape[0] << " " << out.shape[1] << "\n" << flush;
        return out;
    }
};