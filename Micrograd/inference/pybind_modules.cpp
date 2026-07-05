#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "inference_tensor.h"
#include "conv.h"
#include "pool_layers.h"
#include "resnet.h"

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


PYBIND11_MODULE(conv_cpp, m) {
    py::class_<ConvLayer>(m, "ConvLayer")
        .def(py::init<int, int, int, int, int>())
        .def("set_values", &ConvLayer::set_values)
        .def("__call__", &ConvLayer::operator())
        .def("__repr__", &ConvLayer::get_string)
        ;
}

PYBIND11_MODULE(pool_layers, m) {
    py::class_<MaxPoolLayer>(m, "MaxPoolLayer")
        .def(py::init<int, int, int>())
        .def("__call__", &MaxPoolLayer::operator())
        ;
    py::class_<AdaptiveAvgPoolLayer>(m, "AdaptiveAvgPoolLayer")
        .def(py::init<int>())
        .def("__call__", &AdaptiveAvgPoolLayer::operator())
        ;
    py::class_<BN_layer>(m, "BN_layer")
        .def(py::init<size_t>())
        .def("__call__", &BN_layer::operator())
        .def_readwrite("weight", &BN_layer::weight)
        .def_readwrite("bias", &BN_layer::bias)
        .def_readwrite("running_mean", &BN_layer::running_mean)
        .def_readwrite("running_var", &BN_layer::running_var)
        ;
}

PYBIND11_MODULE(resnet, m) {
    py::class_<Stem>(m, "Stem")
        .def(py::init<int, int, int, int, int, int, int, int>())
        .def("__call__", &Stem::operator())
        .def_readwrite("conv", &Stem::conv)
        .def_readwrite("bn_layer", &Stem::bn_layer)
        .def_readwrite("pool", &Stem::pool)
        ;
    
    py::class_<BasicBlock>(m, "BasicBlock")
        .def(py::init<size_t, size_t, size_t>())
        .def("__call__", &BasicBlock::operator())
        .def_readwrite("conv1", &BasicBlock::conv1)
        .def_readwrite("conv2", &BasicBlock::conv2)
        .def_readwrite("conv3", &BasicBlock::conv3)
        .def_readwrite("bn_layers", &BasicBlock::bn_layer)
        ;
    
    py::class_<ClassifierHead>(m, "ClassifierHead")
        .def(py::init<size_t, size_t>())
        .def("__call__", &ClassifierHead::operator())
        .def_readwrite("pool", &ClassifierHead::pool)
        .def_readwrite("FC", &ClassifierHead::FC)
        ;
    
    py::class_<ResNet18>(m, "ResNet18")
        .def(py::init<>())
        .def("__call__", &ResNet18::operator())
        .def_readwrite("stem", &ResNet18::stem)
        .def_readwrite("blocks", &ResNet18::blocks)
        .def_readwrite("classifier_head", &ResNet18::classifier_head)
        ;
}
