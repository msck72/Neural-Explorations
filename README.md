# Neural-Explorations

**Neural-Explorations** is a personal sandbox for experimenting with neural networks and understanding how modern deep learning systems work from the ground up. This repository focuses on building neural network components from scratch, exploring automatic differentiation, and gradually extending toward more advanced abstractions such as tensors and multilayer perceptrons.
---

## Micrograd

This repository includes a custom implementation of **Micrograd**, inspired by Andrej Karpathy’s [Micrograd](https://github.com/karpathy/micrograd) project. The original idea has been extended with additional functionality, cleaner abstractions, and support for more complex use cases.

At its core, Micrograd enables automatic differentiation on scalar values by building a computation graph and applying reverse-mode autodiff (backpropagation).

### Key Enhancements
- Extended scalar-based autograd engine
- Introduced TensorEngine that supports creation of tensors similar to pytorch 
- Support for building multi-layer perceptrons (MLPs)  
- Visualization of computation graphs  

---

## Working with Scalar Values and Automatic Differentiation

Below is a simple example demonstrating how scalar values are wrapped inside `Value` objects, combined through mathematical operations, and differentiated automatically using backpropagation.


```python
from Micrograd.engine import Value
from Micrograd.utils import draw_graph
a = Value(2.0)
b = Value(3.0)

c = a + b
d = c + 4
e = d / c

e.backward()

draw_graph(e)
```

This builds a computation graph dynamically and computes gradients for all intermediate values once .backward() is called.

![Graph of the above expression along with the gradients obtained after backpropagation.](./assets/output.svg)

### Creating a MLP using Values and micrograd engine for backpropagation

Using the scalar-based engine, a simple feedforward neural network can be constructed and trained.
```python
from Micrograd.nn import MLP, mse_loss
from Micrograd.nn import Optimizer

layers = [2, 1]
inputs = [2.0, 3.0]
targets = [1.0]

# model initialization
my_model = MLP(layers)

optimizer = Optimizer(my_model.parameters(), lr=0.01)

# forward pass
output = my_model(inputs)
print(f'model_output: {output}')

optimizer.zero_grad()
loss = mse_loss(output, targets)
print(f'model_loss: {loss.data}')

# backward pass of loss
loss.backward()

my_model.print_grad()
```

### Comparison with PyTorch

An example comparing the model output, loss, and gradients with the PyTorch implementation can be found here:  
[pytorch_micrograd_mlp_comparision.py](http://github.com/msck72/Neural-Explorations/blob/main/Micrograd/examples/pytorch_micrograd_mlp_comparision.py)


### Tensor Engine Extension
Beyond scalar values, this project introduces a Tensor Engine, enabling operations on multidimensional arrays similar to PyTorch tensors. This significantly enhances flexibility and allows construction of more realistic neural networks.

A quick start to tensors using micrograd tensor-engine
```python
from Micrograd.tensor_engine import Tensor

a = Tensor((1, 2))
b = Tensor((1, 2))

c = a + b

print(f'c = {c}')

c.backward()

print(f'a.grad = {a.grad}')
print(f'b.grad = {b.grad}')
print(f'c.grad = {c.grad}')

```

## Building an MLP Using the Tensor Engine
An example showing how to construct and train an MLP using tensor-based operations:
```python
from Micrograd.nn_with_tensors import TensorEngineMLP, TensorEngineOptimizer
from Micrograd.tensor_utils import value_tensor
from Micrograd.utils import format_tensor


input_tensor = value_tensor(1.0, (1, 2))
target_tensor = value_tensor(0.0, (1, 1))

model = TensorEngineMLP([2, 4, 4, 1])
output = model(input_tensor)

print("Model output:", output)

loss = (output - target_tensor) * (output - target_tensor)
print("Loss:", loss)

optimizer = TensorEngineOptimizer(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()

print("\nGradients after backward:")
for layer, param in model.params_dict().items():
    print(f"{layer}: {format_tensor(param.grad)}")

```

### Comparison with PyTorch

Comparison between the tensor-based MLP and an equivalent PyTorch implementation is available here: 
[pytorch_micrograd_tensor_mlp_comparison.py](https://github.com/msck72/Neural-Explorations/blob/main/Micrograd/examples/pytorch_micrograd_tensor_mlp_comparison.py)