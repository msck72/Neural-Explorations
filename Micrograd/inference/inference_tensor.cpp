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


struct InferenceTensor {
    vector<double> data;
    vector<size_t> shape;
    vector<size_t> strides;

    InferenceTensor(vector<size_t> shape) : shape(shape) {
        int total = 1;
        for(int i = 0; i < shape.size(); i++){
            total *= shape[i];
        }
        
        data.resize(total);

        mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);

        for(auto &v: data){
            v = dist(rng);
        }

        compute_strides();
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

    InferenceTensor operator+(const InferenceTensor& o) const {
        return apply(o, [](double a, double b){ return a + b; }); 
    }
    InferenceTensor operator-(const InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a - b; }); 
    }
    InferenceTensor operator*(const InferenceTensor& o) const { 
        return apply(o, [](double a, double b){ return a * b; }); 
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

    void print() const {
        cout << "InferenceTensor(shape=[";
        for (size_t i = 0; i < shape.size(); ++i)
            cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
        cout << "], data=[\n";

        if (shape.size() == 1) {
            cout << "  ";
            for (size_t i = 0; i < data.size(); ++i)
                cout << data[i] << (i + 1 < data.size() ? ", " : "");
            cout << "\n";
        } else if (shape.size() == 2) {
            size_t cols = shape[1];
            for (size_t i = 0; i < shape[0]; ++i) {
                cout << "  ";
                for (size_t j = 0; j < cols; ++j)
                    cout << data[i * cols + j] << (j + 1 < cols ? ", " : "");
                cout << "\n";
            }
        } else {
            for (auto v : data) cout << "  " << v << "\n";
        }
        cout << "])\n";
    }
};

PYBIND11_MODULE(inference_tensor, m) {
    py::class_<InferenceTensor>(m, "InferenceTensor")
        .def(py::init<vector<size_t>>())
        .def("matmul", &InferenceTensor::matmul)
        .def("tanh", &InferenceTensor::tanh)
        .def("transpose", &InferenceTensor::transpose)
        .def("print", &InferenceTensor::print)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def_readonly("data", &InferenceTensor::data)
        .def_readonly("shape", &InferenceTensor::shape)
        .def_readonly("strides", &InferenceTensor::strides);
}

int main() {
    InferenceTensor a({4}), b({4});
    cout << "1D Add: \n";
    (a + b).print();

    InferenceTensor m1({2, 3}), m2({2, 3});
    cout << "\n2D Elementwise Mul:\n";
    (m1 * m2).print();

    InferenceTensor lhs({2, 3}), rhs({3, 2});
    cout << "\nMatmul (2x3) @ (3x2):\n";
    lhs.matmul(rhs).print();

    cout << "\nTanh: \n";
    lhs.tanh().print();

    cout << "\nTranspose: \n";
    lhs.transpose().print();

    return 0;
}