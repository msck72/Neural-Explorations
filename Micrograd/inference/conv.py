from typing import List
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from inference_tensor import InferenceTensor

class Conv2D:
    def __init__(self, filter_size, value=None):
        self.filter_size = filter_size
        if value is not None:
            self.filter = InferenceTensor((filter_size, filter_size), value)
        else:
            self.filter = InferenceTensor((filter_size, filter_size))
        
    def set_filter(self, filter: np.ndarray):
        assert filter.shape == (self.filter_size, self.filter_size), "Filter size mismatch"
        self.filter.set_values(filter)

    def __call__(self, input_tensor, **kwargs):
        
        assert len(input_tensor.shape) == 2, "The input tensor dimension must be 2 while applying conv 2D"

        input_x, input_y = input_tensor.shape

        output_tensor = InferenceTensor((input_x - self.filter_size + 1, input_y - self.filter_size + 1))

        if ("use_threads" in kwargs) and kwargs["use_threads"]:

            def _convolve(i, j):
                temp = 0
                for m in range(self.filter_size):
                    for n in range(self.filter_size):
                        temp += self.filter[m, n] * input_tensor[i + m, j + n]
                output_tensor[i, j] = temp


            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(input_x - self.filter_size + 1):
                    for j in range(input_y - self.filter_size + 1):
                        futures.append(executor.submit(_convolve, i, j))
        
        # output_tensor.print()
        else:
            for i in range(input_x - self.filter_size + 1):
                for j in range(input_y - self.filter_size + 1):
                    temp = 0
                    for m in range(self.filter_size):
                        for n in range(self.filter_size):
                            temp += self.filter[m, n] * input_tensor[i + m, j + n]
                    output_tensor[i, j] = temp

        return output_tensor

class Conv3D:
    def __init__(self, filter_size, depth, value=None):
        self.filter_size = filter_size
        self.depth = depth
        if value is not None:
            self.filter = InferenceTensor((depth, filter_size, filter_size), value)
        else:
            self.filter = InferenceTensor((depth, filter_size, filter_size))
        
    def set_filter(self, filter: np.ndarray):
        assert filter.shape == (self.depth, self.filter_size, self.filter_size), "Filter size mismatch"
        self.filter.set_values(filter)
    
    def __call__(self, input_tensor, **kwargs):
        assert len(input_tensor.shape) == 3, "The input tensor dimension must be 3 while applying conv 3D"
        input_depth, input_x, input_y = input_tensor.shape
        assert input_depth == self.depth, "Input depth must match filter depth"

        output_tensor = InferenceTensor((input_x - self.filter_size + 1, input_y - self.filter_size + 1))

        if ("use_threads" in kwargs) and kwargs["use_threads"]:

            def _convolve(i, j):
                temp = 0
                for d in range(self.depth):
                    for m in range(self.filter_size):
                        for n in range(self.filter_size):
                            temp += self.filter[d, m, n] * input_tensor[d, i + m, j + n]
                output_tensor[i, j] = temp


            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(input_x - self.filter_size + 1):
                    for j in range(input_y - self.filter_size + 1):
                        futures.append(executor.submit(_convolve, i, j))
        
        else:
            for i in range(input_x - self.filter_size + 1):
                for j in range(input_y - self.filter_size + 1):
                    temp = 0
                    for d in range(self.depth):
                        for m in range(self.filter_size):
                            for n in range(self.filter_size):
                                temp += self.filter[d, m, n] * input_tensor[d, i + m, j + n]
                    output_tensor[i, j] = temp

        return output_tensor


class ConvLayer:
    def __init__(self, num_layers, filter_size, depth, stride = 1, padding = 0):
        self.filter_size = filter_size
        self.num_layers = num_layers
        self.depth = depth
        self.stride = stride
        self.padding = padding
        self.conv_filters = [Conv3D(filter_size, depth) for _ in range(num_layers)]
    
    def set_values(self, filters: List[np.ndarray]):
        assert len(filters) == self.num_layers, "Number of filters must match number of layers"
        for i in range(self.num_layers):
            self.conv_filters[i].set_filter(filters[i])

    def __call__(self, input_tensor):
        # avoidin calling conv call for each filter seperately as that is inefficient
        
        if self.padding > 0:
            input_depth, input_x, input_y = input_tensor.shape
            padded_input = InferenceTensor((input_depth, input_x + 2 * self.padding, input_y + 2 * self.padding))
            for d in range(input_depth):
                for i in range(input_x):
                    for j in range(input_y):
                        padded_input[d, i + self.padding, j + self.padding] = input_tensor[d, i, j]
            input_tensor = padded_input
        
        input_depth, input_x, input_y = input_tensor.shape
        output_tensor = InferenceTensor((self.num_layers, input_x - self.filter_size + 1, input_y - self.filter_size + 1))
        def _apply_filter(r, c):
            for layer, _c_f in enumerate(self.conv_filters):
                value = 0
                for i in range(r, r + self.filter_size, self.stride):
                    for j in range(c, c + self.filter_size, self.stride):
                        for k in range(0, self.depth):
                            value += input_tensor[k, i, j] * _c_f.filter[k, i - r, j - c]
                output_tensor[layer, r, c] = value
        
        for row in range(0, input_x - self.filter_size + 1):
            for col in range(0, input_y - self.filter_size + 1):
                _apply_filter(row, col)
        return output_tensor


