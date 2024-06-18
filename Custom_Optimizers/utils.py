import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def Shuffle_MNIST_Data(train_loader, seed):
    torch.manual_seed(seed)
    
    indices = torch.randperm(len(train_loader.dataset))
    
    # Shuffle data and targets using the same indices
    train_loader.dataset.data = train_loader.dataset.data[indices]
    train_loader.dataset.targets = train_loader.dataset.targets[indices]

    return train_loader