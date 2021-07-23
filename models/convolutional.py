import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, filters, kernel_sizes, strides, num_classes):
        super(BaseCNN, self).__init__()
        assert len(filters) == len(kernel_sizes), "Filters length != kernels"
        assert len(kernel_sizes) == len(strides), "Kernels length != strides"

        self.num_classes = num_classes
        self.convolutions = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.convolutions.append(
                nn.Conv2d(
                    filters[i],
                    filters[i+1],
                    kernel_sizes[i],
                    strides[i]
                )
            )
        self.fc = nn.LazyLinear(self.num_classes)
        self.relu = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
            x = self.relu(x)
            x = self.pool(x)

        x = nn.Flatten()(x)
        x = self.fc(x)
        x = nn.Softmax(dim=-1)(x)

        return x
