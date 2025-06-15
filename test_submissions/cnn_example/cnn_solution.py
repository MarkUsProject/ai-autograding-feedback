import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def convolution_2d(image, kernel):
    """Performs a 2D convolution operation on an image with a given kernel."""
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i : i + kernel_h, j : j + kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output


def max_pooling_2d(feature_map, pool_size=(2, 2), stride=2):
    """Applies max pooling operation on a feature map."""
    h, w = feature_map.shape
    pooled_h = (h - pool_size[0]) // stride + 1
    pooled_w = (w - pool_size[1]) // stride + 1

    pooled_output = np.zeros((pooled_h, pooled_w))

    for i in range(0, pooled_h * stride, stride):
        for j in range(0, pooled_w * stride, stride):
            region = feature_map[i : i + pool_size[0], j : j + pool_size[1]]
            pooled_output[i // stride, j // stride] = np.max(region)

    return pooled_output


class CNN(nn.Module):
    """A simple Convolutional Neural Network model using PyTorch."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
