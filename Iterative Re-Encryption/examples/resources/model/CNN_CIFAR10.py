import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)  # 3 input channels (RGB), 32 filters, 5x5 kernel
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # 32 input channels, 64 filters, 5x5 kernel
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the input size for the first fully connected layer
        # CIFAR-10 images are 32x32
        # After first conv (32x32 -> 28x28) and pool (28x28 -> 14x14)
        # After second conv (14x14 -> 10x10) and pool (10x10 -> 5x5)
        # So we have 64 channels of 5x5 feature maps = 64 * 5 * 5
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        # Project header: 2-layer MLP
        self.proj1 = nn.Linear(84, 64)
        self.proj2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 5 * 5)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Project header
        x = F.relu(self.proj1(x))
        x = self.proj2(x)

        return x