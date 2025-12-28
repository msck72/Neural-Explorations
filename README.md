# Neural-Explorations
Neural Experiments is a place where I do some experiments on different deep neural networks!!!

## Micrograd

This repository contains my implementation of Micrograd, inspired by Andrej Karpathy’s [Micrograd](https://github.com/karpathy/micrograd) series. I have extended the original implementation by adding several new features and enhancements.

Currently adding support for scalar values and building MLP using those scalar values.

Tracking gradient using Micrograd engine.

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

![Graph of the above expression along with the gradients obtained after backpropagation.](./assets/output.svg)

### Creating a MLP using Values and micrograd engine for backpropagation

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

### Comparing the model output/loss and gradient with pytorch implementation