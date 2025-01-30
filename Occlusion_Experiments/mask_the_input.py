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

for f in os.listdir('data'):
    if not os.path.isfile(f'./data/{f}'):
        continue
    
    img = Image.open(f'./data/{f}').convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        y_hat = model(img_tensor)

    actual_class = f.split('_')[0]
    print(f'Prediction = {torch.max(y_hat, 1)[1].item()}, actual = {actual_class}')

# plt.imshow(test_tensor.permute(1, 2, 0).numpy())
# plt.show()

# print(f"y_hat[235] = {y_hat[0][235]}")



