#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "inference_tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(inference_tensor, m) {
    py::class_<InferenceTensor>(m, "InferenceTensor")
        .def(py::init<vector<size_t>>())
        .def(py::init<vector<size_t>, double>())
        .def("__matmul__", &InferenceTensor::matmul)
        .def("tanh", &InferenceTensor::tanh)
        .def("transpose", &InferenceTensor::transpose)
        .def("relu", &InferenceTensor::relu)
        .def("print", &InferenceTensor::print)
        .def("__repr__", &InferenceTensor::get_string)
        // .def("__repr__", [](const InferenceTensor &t){
        //     std::string s = "InferenceTensor(shape=[";
        //     for (size_t i = 0; i < t.shape.size(); ++i)
        //         s += std::to_string(t.shape[i]) + (i + 1 < t.shape.size() ? ", " : "");
        //     s += "])";
        //     return s;
        // })
        .def("flat_index", &InferenceTensor::flat_index)
        .def("__get_item__", &InferenceTensor::get_item)
        .def("__setitem__", &InferenceTensor::set_item)
        .def("set_values", &InferenceTensor::set_values)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def_readonly("data", &InferenceTensor::data)
        .def_readonly("shape", &InferenceTensor::shape)
        .def_readonly("strides", &InferenceTensor::strides);
}

// int main() {
//     InferenceTensor a({4}), b({4});
//     cout << "1D Add: \n";
//     (a + b).print();

//     InferenceTensor m1({2, 3}), m2({2, 3});
//     cout << "\n2D Elementwise Mul:\n";
//     (m1 * m2).print();

//     InferenceTensor lhs({2, 3}), rhs({3, 2});
//     cout << "\nMatmul (2x3) @ (3x2):\n";
//     lhs.matmul(rhs).print();

//     cout << "\nTanh: \n";
//     lhs.tanh().print();

//     cout << "\nTranspose: \n";
//     lhs.transpose().print();


//     InferenceTensor t({2, 2}, 1.0);

//     return 0;
// }