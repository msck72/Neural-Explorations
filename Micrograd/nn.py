from Micrograd.engine import Value
import random

class Neuron:
    def __init__(self, in_features):
        random.seed(42)
        self.weights = [Value(random.random()) for i in range(in_features)]
        self.bias = Value(random.random())
    
    def __call__(self, x):
        out = sum([w_i * x_i for w_i, x_i in zip(self.weights, x)], self.bias)
        return out.tanh()
    
    def parameters(self):
        return self.weights + [self.bias]
    

class Layer:
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    


class MLP:
    def __init__(self, layers):
        self.layers = [Layer(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

    def parameters(self):
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                params.extend(neuron.parameters())
        return params

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def __repr__(self):
        model_str = ''
        for i, layer in enumerate(self.layers):
            model_str += f'Layer: {i}\n'
            for neuron in layer.neurons:
                model_str += f'  Neuron: {neuron.weights}, {neuron.bias}\n'
            model_str += '\n'
        return model_str

    def print_grad(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}:')
            for j, neuron in enumerate(layer.neurons):
                weights_grads = [w.grad for w in neuron.weights]
                bias_grad = neuron.bias.grad
                print(f'  Neuron {j} weights grads: {weights_grads}, bias grad: {bias_grad}')
            print()
        return

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0

    def step(self):
        for param in self.parameters:
            param += self.lr * param.grad


def mse_loss(predicted, target):
    assert len(predicted) == len(target), "Length of predicted and target must be the same."
    loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)
    return loss