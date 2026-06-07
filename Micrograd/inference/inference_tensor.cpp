#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

using namespace std;
namespace py = pybind11;


struct _InferenceTensor {
    vector<double> data;
    vector<size_t> shape;
    vector<size_t> strides;

    _InferenceTensor(vector<size_t> shape) : shape(shape) {
        int total = set_shape(shape);
        
        data.resize(total);

        mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);

        for(auto &v: data){
            v = dist(rng);
        }

        compute_strides();
    }

    _InferenceTensor(vector<size_t> shape, double value) : shape(shape) {
        int total = set_shape(shape);
        
        data.resize(total, value);
        compute_strides();
    }

    int set_shape(vector<size_t> new_shape) {
        int total = 1;
        for(int i = 0; i < shape.size(); i++){
            total *= shape[i];
        }
        return total;
    }

    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    size_t flat_index(const vector<size_t>& idx) const {
        size_t flat = 0;
        for(int i = 0; i < idx.size(); i++){
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

    _InferenceTensor apply(const _InferenceTensor& other, function<double(double, double)> op) const {
        if (shape != other.shape) throw runtime_error("Shape mismatch");
        _InferenceTensor out(shape);
        for (size_t i = 0; i < data.size(); ++i)
            out.data[i] = op(data[i], other.data[i]);
        return out;
    }

    _InferenceTensor apply(function<double(double)> op) const {
        _InferenceTensor out(shape);
        for (size_t i = 0; i < data.size(); ++i)
            out.data[i] = op(data[i]);
        return out;
    }

    _InferenceTensor operator+(const _InferenceTensor& o) const {
        return apply(o, [](double a, double b){ return a + b; }); 
    }
    _InferenceTensor operator-(const _InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a - b; }); 
    }
    _InferenceTensor operator*(const _InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a * b; }); 
    }

    _InferenceTensor matmul(const _InferenceTensor& other) const {
        if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0])
            throw runtime_error("Invalid shapes for matmul");

        size_t M = shape[0], K = shape[1], N = other.shape[1];
        _InferenceTensor out({M, N});
        fill(out.data.begin(), out.data.end(), 0.0);

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < K; ++k)
                    out.data[i * N + j] += data[i * K + k] * other.data[k * N + j];
        return out;
    }

    _InferenceTensor tanh() const {
        return apply([](double x){ return ::tanh(x); });
    }

    _InferenceTensor transpose() const {
        if (shape.size() != 2) throw runtime_error("Transpose only for 2D _InferenceTensors");
        size_t rows = shape[0], cols = shape[1];
        _InferenceTensor out({cols, rows});
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out.data[j * rows + i] = data[i * cols + j];
        return out;
    }

    _InferenceTensor relu() const {
        return apply([](double x){ return x > 0 ? x : 0; });
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
        cout << "_InferenceTensor(shape=[";
        for (size_t i = 0; i < shape.size(); ++i)
            cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
        cout << "])\n";
        print_rec(0, 0, 0);
    }
};

PYBIND11_MODULE(inference_tensor_cpp, m) {
    py::class_<_InferenceTensor>(m, "_InferenceTensor")
        .def(py::init<vector<size_t>>())
        .def(py::init<vector<size_t>, double>())
        .def("matmul", &_InferenceTensor::matmul)
        .def("tanh", &_InferenceTensor::tanh)
        .def("transpose", &_InferenceTensor::transpose)
        .def("relu", &_InferenceTensor::relu)
        .def("print", &_InferenceTensor::print)
        .def("flat_index", &_InferenceTensor::flat_index)
        .def("get_item", &_InferenceTensor::get_item)
        .def("set_item", &_InferenceTensor::set_item)
        .def("set_values", &_InferenceTensor::set_values)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def_readonly("data", &_InferenceTensor::data)
        .def_readonly("shape", &_InferenceTensor::shape)
        .def_readonly("strides", &_InferenceTensor::strides);
}

int main() {
    _InferenceTensor a({4}), b({4});
    cout << "1D Add: \n";
    (a + b).print();

    _InferenceTensor m1({2, 3}), m2({2, 3});
    cout << "\n2D Elementwise Mul:\n";
    (m1 * m2).print();

    _InferenceTensor lhs({2, 3}), rhs({3, 2});
    cout << "\nMatmul (2x3) @ (3x2):\n";
    lhs.matmul(rhs).print();

    cout << "\nTanh: \n";
    lhs.tanh().print();

    cout << "\nTranspose: \n";
    lhs.transpose().print();


    _InferenceTensor t({2, 2}, 1.0);

    return 0;
}