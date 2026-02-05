import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()

        # -------- Convolutional layers --------
        # Input: (N, 3, 32, 32)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (N, 16, 16, 16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (N, 32, 8, 8)

        # -------- Fully connected layers --------
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.tanh3 = nn.Tanh()

        self.fc2 = nn.Linear(120, 100)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Convolution block 1
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)

        # Convolution block 2
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.tanh3(x)
        x = self.fc2(x)

        return x
