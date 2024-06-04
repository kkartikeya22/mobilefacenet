import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn import Parameter

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(ComplexConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        real = self.real_conv(x[:, 0:1]) - self.imag_conv(x[:, 1:2])
        imag = self.real_conv(x[:, 1:2]) + self.imag_conv(x[:, 0:1])
        return torch.stack([real, imag], dim=1)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    @property
    def weight(self):
        return torch.stack([self.real_bn.weight, self.imag_bn.weight], dim=0)

    @weight.setter
    def weight(self, value):
        self.real_bn.weight.data = value[0].data
        self.imag_bn.weight.data = value[1].data

    @property
    def bias(self):
        return torch.stack([self.real_bn.bias, self.imag_bn.bias], dim=0)

    @bias.setter
    def bias(self, value):
        self.real_bn.bias.data = value[0].data
        self.imag_bn.bias.data = value[1].data

    def forward(self, x):
        real = self.real_bn(x[:, 0])
        imag = self.imag_bn(x[:, 1])
        return torch.stack([real, imag], dim=1)

# Define other complex layers and model components as before...

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(50, 2, 112, 96))  # Adjusted input size to match the expected input
    net = ComplexMobileFacenet(bottleneck_setting=[(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1)])
    print(net)
    x = net(input)
    print(x.shape)
