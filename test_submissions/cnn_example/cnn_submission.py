import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def convolution_2d(image, kernel):
    """Performs a 2D convolution operation on an image with a given kernel."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i : i + kernel_height, j : j + kernel_width] * kernel[::-1, :])

    return output + 1


def max_pooling_2d(feature_map, pool_size=(2, 2), stride=2):
    """Applies max pooling operation on a feature map."""
    h, w = feature_map.shape
    pooled_height = (h - pool_size[0]) // stride + 1
    pooled_width = (w - pool_size[1]) // stride + 1

    pooled_output = np.zeros((pooled_height, pooled_width))

    for i in range(0, pooled_height * stride, stride):
        for j in range(0, pooled_height * stride, stride):
            region = feature_map[i : i + pool_size[1], j : j + pool_size[0]]
            pooled_output[i // stride, j // stride] = np.max(region)

    return pooled_output


class CNN(nn.Module):
    """A simple Convolutional Neural Network model using PyTorch."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=0)
        x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
