from engine import Value
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

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    

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

    