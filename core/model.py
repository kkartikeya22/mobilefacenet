import numpy as np
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
        real = self.real_conv(x[:, 0]) - self.imag_conv(x[:, 1])
        imag = self.real_conv(x[:, 1]) + self.imag_conv(x[:, 0])
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

class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super(ComplexPReLU, self).__init__()
        self.real_prelu = nn.PReLU(num_parameters)
        self.imag_prelu = nn.PReLU(num_parameters)

    def forward(self, x):
        real = self.real_prelu(x[:, 0])
        imag = self.imag_prelu(x[:, 1])
        return torch.stack([real, imag], dim=1)

class ComplexConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dw=False, linear=False):
        super(ComplexConvBlock, self).__init__()
        if dw:
            self.conv = ComplexConv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes)
        else:
            self.conv = ComplexConv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm2d(out_planes)
        self.linear = linear
        if not self.linear:
            self.prelu = nn.PReLU(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if not self.linear:
            out = self.prelu(out)
        return out

class ComplexBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expansion):
        super(ComplexBottleneck, self).__init__()
        planes = in_planes * expansion
        self.conv1 = ComplexConvBlock(in_planes, planes, 1, 1, 0)
        self.conv2 = ComplexConvBlock(planes, planes, 3, stride, 1, dw=True)
        self.conv3 = ComplexConvBlock(planes, out_planes, 1, 1, 0, linear=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = ComplexConvBlock(in_planes, out_planes, 1, 1, 0, linear=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        return out

class ComplexMobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting):
        super(ComplexMobileFacenet, self).__init__()
        self.conv1 = ComplexConvBlock(2, 64, 3, 3, 1)  # Changed input channels from 3 to 2
        self.dw_conv1 = ComplexConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = ComplexBottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ComplexConvBlock(self.inplanes, 512, 1, 1, 0)
        self.linear7 = ComplexConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = ComplexConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                nn.init.kaiming_normal_(m.real_conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.imag_conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, ComplexBatchNorm2d):
                nn.init.constant_(m.real_bn.weight, 1)
                nn.init.constant_(m.real_bn.bias, 0)
                nn.init.constant_(m.imag_bn.weight, 1)
                nn.init.constant_(m.imag_bn.bias, 0)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            out_planes = c
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, out_planes, s, t))
                else:
                    layers.append(block(self.inplanes, out_planes, 1, t))
                self.inplanes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dw_conv1(out)
        out = self.blocks(out)
        out = self.conv2(out)
        out = self.linear7(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

class ComplexArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ComplexArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# Preprocessing function to convert 3-channel input to 2-channel complex representation
def preprocess_input(input):
    # Assuming the input has three channels (RGB), we convert it to two channels (real and imaginary parts)
    real_part = input[:, :2, :, :]  # Taking the first two channels as the real part
    imag_part = input[:, 2:3, :, :]  # Taking the third channel as the imaginary part
    complex_input = torch.cat([real_part, imag_part], dim=1)
    return complex_input

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(2, 3, 112, 96))  # Original 3-channel input
    input = preprocess_input(input)  # Convert to 2-channel complex input
    net = ComplexMobileFacenet(bottleneck_setting=[(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1)])
    print(net)
    x = net(input)
    print(x.shape)
