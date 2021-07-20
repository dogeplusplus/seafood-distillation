import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, filters, kernel_sizes, strides, num_classes):

        assert len(filters) == len(kernel_sizes), "Filters length != kernels"
        assert len(kernel_sizes) == len(strides), "Kernels length != strides"

        self.num_classes = num_classes
        self.convolutions = []
        for i in range(len(self.filters) - 1):
            self.convolutions.append(
                nn.Conv2d(
                    filters[i],
                    filters[i+1],
                    kernel_sizes[i],
                    strides[i]
                )
            )
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
            x = self.relu(x)
            x = self.pool(x)

        x = self.flatten()
        x = nn.Linear(x.shape[-1], self.num_classes)
        x = nn.Softmax()(x)

        return x
