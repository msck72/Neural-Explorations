from inference_tensor import InferenceTensor

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = InferenceTensor((output_size, input_size))
        self.biases = InferenceTensor((output_size, 1))
    
    def __call__(self, input_tensor):
        print(f"input_tensor.shape[0] = {input_tensor.shape[0]}, self.input_size = {self.input_size}")
        assert input_tensor.shape[0] == self.input_size, "Input tensor shape mismatch"
        output_tensor = self.weights @ input_tensor + self.biases
        return output_tensor

class SimpleNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    
    def __call__(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer(output)
        return output

