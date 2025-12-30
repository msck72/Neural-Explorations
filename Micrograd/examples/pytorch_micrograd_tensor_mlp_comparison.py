import torch

from Micrograd.nn_with_tensors import TensorEngineMLP, TensorEngineOptimizer, PytorchMLP
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

print("\n" + "-" * 30 + "\n")


print("----- PyTorch Model -----")

pytorch_model = PytorchMLP(model)
pytorch_input = torch.full((1, 2), 1.0)

pytorch_output = pytorch_model(pytorch_input)
print("Pytorch model output:", pytorch_output)

pytorch_target = torch.full((1, 1), 0.0)
pytorch_loss = (pytorch_output - pytorch_target) ** 2
print("Pytorch loss:", pytorch_loss)

pytorch_loss.backward()

for name, param in pytorch_model.named_parameters():
    print(f"{name} grad:", param.grad)