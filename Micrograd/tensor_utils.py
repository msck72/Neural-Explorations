from Micrograd.tensor_engine import Tensor

def fill_tensor_with_value(tensor, shape, value):
    if len(shape) == 1:
            for i in range(shape[0]):
                tensor[i] = value
            return
    for i in range(shape[0]):
        fill_tensor_with_value(tensor[i], shape[1:], value)

def ones_tensor(shape):
    tensor = Tensor(shape)
    fill_tensor_with_value(tensor.tensor, shape, 1.0)
    return tensor

def zeros_tensor(shape):
    tensor = Tensor(shape)
    fill_tensor_with_value(tensor.tensor, shape, 0.0)
    return tensor

def value_tensor(value, shape):
    tensor = Tensor(shape)
    fill_tensor_with_value(tensor.tensor, shape, float(value))
    return tensor