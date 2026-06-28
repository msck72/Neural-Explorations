#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "inference_tensor.h"

using namespace std;
namespace py = pybind11;

struct ConvLayer{
    size_t num_layers;
    size_t filter_size;
    size_t depth;
    size_t stride;
    size_t padding;

    vector<_InferenceTensor> conv_filters;

    ConvLayer(int num_layers, int filter_size, int depth, int stride = 1, int padding = 0)
        : num_layers(num_layers), filter_size(filter_size), depth(depth), stride(stride), padding(padding) {
        conv_filters.reserve(this->num_layers);
        for(size_t i = 0; i < this->num_layers; i++){
            conv_filters.emplace_back(vector<size_t>{this->depth, this->filter_size, this->filter_size});
        }
    }

    void set_values(const vector<vector<vector<vector<double>>>>& filters){
        assert(filters.size() == num_layers);
        for(size_t i = 0; i < num_layers; i++){
            assert(filters[i].size() == depth);
            vector<double> flat_values;
            flat_values.reserve(depth * filter_size * filter_size);
            for(size_t d = 0; d < depth; d++){
                assert(filters[i][d].size() == filter_size);
                for(size_t x = 0; x < filter_size; x++){
                    assert(filters[i][d][x].size() == filter_size);
                    for(size_t y = 0; y < filter_size; y++){
                        flat_values.push_back(filters[i][d][x][y]);
                    }
                }
            }
            conv_filters[i].set_values(flat_values);
        }
    }

    _InferenceTensor operator()(const _InferenceTensor& input_tensor){
        _InferenceTensor padded_input = input_tensor;
        if(padding > 0){
            size_t input_depth = input_tensor.shape[0];
            size_t input_x = input_tensor.shape[1];
            size_t input_y = input_tensor.shape[2];
            padded_input = _InferenceTensor({input_depth, input_x + 2 * padding, input_y + 2 * padding});
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

        _InferenceTensor output_tensor({num_layers, (input_x - filter_size) / stride + 1, (input_y - filter_size) / stride + 1});

        auto _apply_filter = [&](size_t r, size_t c){
            for(size_t layer = 0; layer < num_layers; layer++){
                double value = 0;
                for(size_t i = r; i < r + filter_size; i++){
                    for(size_t j = c; j < c + filter_size; j++){
                        for(size_t k = 0; k < depth; k++){
                            value += padded_input.get_item({k, i, j}) * conv_filters[layer].get_item({k, i - r, j - c});
                        }
                    }
                }
                output_tensor.set_item({layer, r / stride, c / stride}, value);
            }
        };

        for(size_t row = 0; row <= input_x - filter_size; row += stride){
            vector<thread> threads;
            for(size_t col = 0; col <= input_y - filter_size; col += stride){
                threads.emplace_back(_apply_filter, row, col);
            }
            for(auto& t : threads){
                t.join();
            }
        }
        return output_tensor;
    }
};


PYBIND11_MODULE(conv_cpp, m) {
    py::class_<ConvLayer>(m, "ConvLayer")
        .def(py::init<int, int, int, int, int>())
        .def("set_values", &ConvLayer::set_values)
        .def("__call__", &ConvLayer::operator());
}