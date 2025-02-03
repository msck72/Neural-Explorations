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
target_classes = []
file_names = []

for f in os.listdir('data'):
    if not os.path.isfile(f'./data/{f}'):
        continue
    
    file_names.append(f.split('.')[0])

    img = Image.open(f'./data/{f}').convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    if img_stack is None:
        img_stack = img_tensor
    else:
        img_stack = torch.cat((img_stack, img_tensor), dim=0)
    
    target_classes.append(int(f.split('_')[0]))
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_classes = torch.Tensor(target_classes).to(device).type(torch.int64)
target_classes = target_classes.view((10, 1))
num_images, _, height, width = img_stack.size()


filter_size = 100
filter_vals = torch.zeros((num_images, 3, filter_size, filter_size))
mask = torch.ones_like(img_stack)
mask[:, :, :filter_size, :filter_size] = filter_vals

model = model.to(device)
img_stack = img_stack.to(device)
mask = mask.to(device)

prob_variations = torch.zeros((num_images, height - filter_size + 1, width - filter_size + 1))
prob_variations = prob_variations.to(device)

for i in range(height - filter_size + 1):
    moving_filter = mask.clone()
    moving_filter = torch.roll(moving_filter, i, dims=(2))
    for j in range(width - filter_size):
        
        with torch.no_grad():
            y_hat = model(img_stack * moving_filter)
            prob_variations[:, i, j] = y_hat.gather(1, target_classes).squeeze(1)
            # print(f'Prediction = {torch.max(y_hat, 1)[1]}')
            moving_filter = torch.roll(moving_filter, 1, dims=(3))
    print("one hroizontal level completed")

os.makedirs('./probability_variations', exist_ok=True)

for i in range(target_classes.size(0)):
    torch.save(prob_variations[i], f'./probability_variations/{file_names[i]}.pth')

# print(prob_variations, end="\n\n\n")


# img = Image.open(f'./data/200_1.jpeg').convert('RGB')
# img_tensor = transform(img)
# img_tensor = img_tensor.unsqueeze(0)

# filter_size = 10
# filter = torch.ones((1, 3, filter_size, filter_size))

# mask = torch.zeros_like(img_tensor)
# mask[:, :, :filter_size, :filter_size] = filter
# mask = 1 - mask


# model = resnet18(weights="DEFAULT")
# model.eval()

# with torch.no_grad():
#     ans_before = model(img_tensor)
#     print(f"ans _ before = {torch.max(ans_before, 1)[1]}")

# img_tensor = img_tensor * mask

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