import torch
import torchvision.transforms as transforms
import os

from PIL import Image
import matplotlib.pyplot as plt


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


for f in os.listdir('./data'):
    img = Image.open(f'./data/{f}')

    img_tensor = transform(img)
    
    fig, axes = plt.subplots(1, 2)
    
    axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axes[1].imshow(torch.load(f'./probability_variations/{f.split(".")[0]}.pth').cpu().numpy(), cmap='gray')
    plt.show()