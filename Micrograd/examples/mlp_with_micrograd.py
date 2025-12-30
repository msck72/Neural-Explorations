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