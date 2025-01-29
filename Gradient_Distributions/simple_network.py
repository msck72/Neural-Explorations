import torch
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os

def download_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = MNIST(root="./", train=True, download=True, transform=transform)
    testset=MNIST(root="./", train=False, download=True, transform=transform)

    return trainset, testset

def create_dataloaders(trainset, testset, batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader



class SimpleNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer_1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x



def train(model, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train(True)

    for epoch in range(epochs):
        batch_num = 0
        for images, targets in trainloader:
            y_hat = model(images)
            loss = criterion(y_hat, targets)

            optimizer.zero_grad()
            loss.backward()

            os.makedirs('./Gradients', exist_ok=True)
            if batch_num % 20 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        os.makedirs(f'./Gradients/{name}', exist_ok=True)
                        grad_file = f"Gradients/{name}/{batch_num}.pt"
                        torch.save(param.grad.clone(), grad_file)

            optimizer.step()
            batch_num += 1



def test(model, testloader):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in testloader:

            y_hat = model(images)

            loss = criterion(y_hat, targets)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(y_hat, 1)

            total_correct += (predicted == targets).sum().item()

            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():

    train_set, test_set = download_mnist()
    train_loader, test_loader = create_dataloaders(train_set, test_set, 256)

    input_dim = 28 * 28

    model = SimpleNetwork(input_dim)

    train(model, train_loader, 1)

    test_loss, test_accuracy = test(model, test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()