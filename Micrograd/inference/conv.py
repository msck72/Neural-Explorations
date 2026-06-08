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

