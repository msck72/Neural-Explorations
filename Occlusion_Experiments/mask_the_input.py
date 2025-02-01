import torch
from PIL import Image

from torchvision.transforms import transforms
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = resnet18(weights='DEFAULT')
model.eval()

img_stack = None


for f in os.listdir('data'):
    if not os.path.isfile(f'./data/{f}'):
        continue
    
    img = Image.open(f'./data/{f}').convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    if img_stack is None:
        img_stack = img_tensor
    else:
        img_stack = torch.cat((img_stack, img_tensor), dim=0)
    

num_images, _, height, width = img_tensor.size()

filter_size = 100
filter_vals = torch.zeros((num_images, 3, filter_size, filter_size))
actual_filter = torch.ones_like(img_stack)
actual_filter[:, :, :filter_size, :filter_size] = filter_vals

# prob_variations = torch.zeros((height - filter_size + 1, width - filter_size + 1))

for i in range(height - filter_size + 1):
    moving_filter = actual_filter.clone()
    moving_filter = torch.roll(moving_filter, i, dims=(2))
    for j in range(width - filter_size):
    
        with torch.no_grad():
            y_hat = model(img_stack * moving_filter)
            print(f'Prediction = {torch.max(y_hat, 1)[1]}')
            moving_filter = torch.roll(moving_filter, 1, dims=(3))
    print("one hroizontal level completed")

# print(prob_variations, end="\n\n\n")


# img = Image.open(f'./data/200_1.jpeg').convert('RGB')
# img_tensor = transform(img)
# img_tensor = img_tensor.unsqueeze(0)

# filter_size = 10
# filter = torch.ones((1, 3, filter_size, filter_size))

# actual_filter = torch.zeros_like(img_tensor)
# actual_filter[:, :, :filter_size, :filter_size] = filter
# actual_filter = 1 - actual_filter


# model = resnet18(weights="DEFAULT")
# model.eval()

# with torch.no_grad():
#     ans_before = model(img_tensor)
#     print(f"ans _ before = {torch.max(ans_before, 1)[1]}")

# img_tensor = img_tensor * actual_filter

# with torch.no_grad():
#     ans = model(img_tensor)
#     print(f"ans = {torch.max(ans, 1)[1]}")
# img_tensor = img_tensor.squeeze(0)
# # print(f"{img_tensor.size(0)}")

# plt.imshow(img_tensor.permute(1, 2, 0).numpy())
# plt.show()



# # plt.imshow(test_tensor.permute(1, 2, 0).numpy())
# # plt.show()

# # print(f"y_hat[235] = {y_hat[0][235]}")