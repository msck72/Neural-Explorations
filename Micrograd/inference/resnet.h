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

    pair<InferenceTensor, map<string, InferenceTensor>> operator()(const InferenceTensor& input_tensor){
        cout << "Entering ResNet18 operator()\n" << flush;
        cout << "Input tensor shape: " << input_tensor.shape[0] << " " << input_tensor.shape[1] << " " << input_tensor.shape[2] << "\n" << flush;
        
        cout << "Calling stem...\n" << flush;
        map<string, InferenceTensor> activations;  // To store intermediate activations
        InferenceTensor out = stem(input_tensor);
        activations.emplace("stem", out);  // Store the output of the stem
        
        int cntr = 0;
        for(auto& block : blocks){
            if(cntr == 0){
                out = block(out, activations, "block_0");
            }
            else if(cntr == 1){
                out = block(out, activations, "block_1");
            }
            else{
                out = block(out);
            }
            activations.emplace("block_" + to_string(cntr), out);  // Store the output of each block
            cntr++;
        }
        cout << "All blocks done. Calling classifier head...\n" << flush;
        out = classifier_head(out, activations);
        activations.emplace("fc_output", out);  // Store the output of the classifier head
        cout << "ResNet18 forward pass complete\n" << flush;
        return {out, activations};  // Return both the final output and the activations
    }

};