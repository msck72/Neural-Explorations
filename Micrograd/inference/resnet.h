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
#include "blocks.h"
#include "pool_layers.h"

using namespace std;

struct ResNet18{
    Stem stem;
    vector<BasicBlock> blocks;
    ClassifierHead classifier_head;
    ResNet18():stem(3, 64, 7, 2, 3, 3, 2, 1), classifier_head(512, 1000){
        blocks.emplace_back(BasicBlock(64, 64, 1));
        blocks.emplace_back(BasicBlock(64, 64, 1));
        blocks.emplace_back(BasicBlock(64, 128, 2));
        blocks.emplace_back(BasicBlock(128, 128, 1));
        blocks.emplace_back(BasicBlock(128, 256, 2));
        blocks.emplace_back(BasicBlock(256, 256, 1));
        blocks.emplace_back(BasicBlock(256, 512, 2));
        blocks.emplace_back(BasicBlock(512, 512, 1));
    }

    InferenceTensor operator()(const InferenceTensor& input_tensor){
        InferenceTensor out = input_tensor;
        for(auto& block : blocks){
            out = block(out);
        }
        return out;
    }

};