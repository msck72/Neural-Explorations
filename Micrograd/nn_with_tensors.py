import random

import torch
import torch.nn as nn

from Micrograd.tensor_engine import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor((in_features, out_features))
        self.weights.from_values(tensor=[[random.uniform(-1, 1) for _ in range(out_features)] for _ in range(in_features)],
                                 formed_by=(),
                                 op='init')
        self.bias = Tensor((1, out_features))
        self.bias.from_values(tensor=[[random.uniform(-1, 1) for _ in range(out_features)]],
                             formed_by=(),
                             op='init')

    def __call__(self, x):
        return x @ self.weights + self.bias

class TensorEngineMLP:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Linear(layers[i], layers[i + 1]))

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.weights)
            params.append(layer.bias)
        return params

    def params_dict(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layer_{i}_weights'] = layer.weights
            params[f'layer_{i}_bias'] = layer.bias
        return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = x.tanh()
        return x

class PytorchMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()

        self.layers = nn.ModuleList()
        for layer in mlp.layers:
            linear = nn.Linear(
                in_features=layer.weights.shape[0],
                out_features=layer.weights.shape[1],
                bias=True
            )

            linear.weight.data = torch.tensor(layer.weights.tensor).T
            linear.bias.data = torch.tensor(layer.bias.tensor)

            self.layers.append(linear)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.tanh(x)
        return x
        

class TensorEngineOptimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.tensor -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = [[0.0 for _ in range(len(param.grad[0]))] for _ in range(len(param.grad))]