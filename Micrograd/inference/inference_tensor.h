// Header for InferenceTensor used by inference_tensor.cpp
#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <sstream>

using namespace std;

struct InferenceTensor {
    vector<double> data;
    vector<size_t> shape;
    vector<size_t> strides;

    InferenceTensor(vector<size_t> shape) : shape(shape) {
        int total = set_shape(shape);
        
        data.resize(total);

        mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);

        for(auto &v: data){
            v = dist(rng);
        }

        compute_strides();
    }

    InferenceTensor(vector<size_t> shape, double value) : shape(shape) {
        int total = set_shape(shape);
        
        data.resize(total, value);
        compute_strides();
    }

    InferenceTensor(vector<size_t> shape, const vector<double>& values) : shape(shape) {
        int total = set_shape(shape);
        
        if (values.size() != (int) total)
            throw runtime_error("Size mismatch");
        data = values;
        compute_strides();
    }

    int set_shape(const vector<size_t>& new_shape) {
        int total = 1;
        for(size_t i = 0; i < new_shape.size(); i++){
            total *= new_shape[i];
        }
        return total;
    }

    void compute_strides() {
        strides.resize(shape.size());
        if (shape.empty()) return;
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    size_t flat_index(const vector<size_t>& idx) const {
        size_t flat = 0;
        for(size_t i = 0; i < idx.size(); i++){
            flat += idx[i] * strides[i];
        }
        return flat;
    }

    double get_item(const vector<size_t>& idx) const {
        return data[flat_index(idx)];
    }

    void set_item(const vector<size_t>& idx, double value) {
        data[flat_index(idx)] = value;
    }

    void set_values(const vector<double>& flat_values) {
        if (flat_values.size() != data.size())
            throw runtime_error("Size mismatch");
        data = flat_values;
    }

    InferenceTensor apply(const InferenceTensor& other, function<double(double, double)> op) const {
        if (shape != other.shape) throw runtime_error("Shape mismatch");
        InferenceTensor out(shape);
        for (size_t i = 0; i < data.size(); ++i)
            out.data[i] = op(data[i], other.data[i]);
        return out;
    }

    InferenceTensor apply(function<double(double)> op) const {
        InferenceTensor out(shape);
        for (size_t i = 0; i < data.size(); ++i)
            out.data[i] = op(data[i]);
        return out;
    }

    void apply_inplace(const InferenceTensor& other, function<double(double, double)> op) {
        for(int i = other.shape.size() - 1; i >= 0; i--){
            if(other.shape[i] != shape[i]){
                throw runtime_error("Shape mismatch");
            }
        }

        int iter_count = 1;
        for(int i = 0; i < shape.size() - other.shape.size(); i++){
            iter_count *= shape[i];
        }

        for(int i = 0; i < iter_count; i++){
            for(int j = 0; j < other.data.size(); j++){
                data[i * other.data.size() + j] = op(data[i * other.data.size() + j], other.data[j]);
            }
        }
    }

    InferenceTensor operator+(const InferenceTensor& o) const {
        return apply(o, [](double a, double b){ return a + b; }); 
    }
    InferenceTensor operator-(const InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a - b; }); 
    }
    InferenceTensor operator*(const InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a * b; }); 
    }

    void add_inplace(const InferenceTensor& o) { 
        apply_inplace(o, [](double a, double b){ return a + b; }); 
    }
    void sub_inplace(const InferenceTensor& o) { 
        apply_inplace(o, [](double a, double b){ return a - b; }); 
    }
    void mul_inplace(const InferenceTensor& o) {    
        apply_inplace(o, [](double a, double b){ return a * b; }); 
    }
    void div_inplace(const InferenceTensor& o) {    
        apply_inplace(o, [](double a, double b){ return a / b; }); 
    }
    void sqrt_inplace() {    
        for(size_t i = 0; i < data.size(); i++){
            data[i] = sqrt(data[i]);
        }
    }

    InferenceTensor matmul(const InferenceTensor& other) const {
        if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0])
            throw runtime_error("Invalid shapes for matmul");

        size_t M = shape[0], K = shape[1], N = other.shape[1];
        InferenceTensor out({M, N});
        fill(out.data.begin(), out.data.end(), 0.0);

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < K; ++k)
                    out.data[i * N + j] += data[i * K + k] * other.data[k * N + j];
        return out;
    }

    InferenceTensor tanh() const {
        return apply([](double x){ return ::tanh(x); });
    }

    InferenceTensor transpose() const {
        if (shape.size() != 2) throw runtime_error("Transpose only for 2D InferenceTensors");
        size_t rows = shape[0], cols = shape[1];
        InferenceTensor out({cols, rows});
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out.data[j * rows + i] = data[i * cols + j];
        return out;
    }

    InferenceTensor relu() const {
        return apply([](double x){ return x > 0 ? x : 0; });
    }

    InferenceTensor flatten() const {
        return InferenceTensor({data.size()}, data);
    }

    void print_rec(int dim, int indentation, int start) const {
        for (int i = 0; i < indentation; i++) cout << " ";

        if (dim == (int)strides.size() - 1) {
            cout << "[";
            for (int i = 0; i < (int)shape[dim]; i++)
                cout << data[start + i] << (i + 1 < (int)shape[dim] ? ", " : "");
            cout << "]\n";
            return;
        }

        cout << "[\n";
        for (int i = 0; i < (int)shape[dim]; i++)
            print_rec(dim + 1, indentation + 2, start + strides[dim] * i);
        for (int i = 0; i < indentation; i++) cout << " ";
        cout << "],\n";
    }

    void print() const {
        cout << "InferenceTensor(shape=[";
        for (size_t i = 0; i < shape.size(); ++i)
            cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
        cout << "])\n";
        print_rec(0, 0, 0);
    }

    void get_string_rec(int dim, int indentation, int start, stringstream& ss) const {
        for (int i = 0; i < indentation; i++) ss << " ";

        if (dim == (int)strides.size() - 1) {
            ss << "[";
            for (int i = 0; i < (int)shape[dim]; i++)
                ss << data[start + i] << (i + 1 < (int)shape[dim] ? ", " : "");
            ss << "]\n";
            return;
        }

        ss << "[\n";
        for (int i = 0; i < (int)shape[dim]; i++)
            get_string_rec(dim + 1, indentation + 2, start + strides[dim] * i, ss);
        for (int i = 0; i < indentation; i++) ss << " ";
        ss << "],\n";
    }

    string get_string() const {
        stringstream ss;
        ss << "InferenceTensor(shape=[";
        for (size_t i = 0; i < shape.size(); ++i)
            ss << shape[i] << (i + 1 < shape.size() ? ", " : "");
        ss << "])\n";
        get_string_rec(0, 0, 0, ss);
        return ss.str();
    }
};
