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
