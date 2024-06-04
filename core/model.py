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

class ComplexBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(ComplexBottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            ComplexConv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            ComplexBatchNorm2d(inp * expansion),
            ComplexPReLU(inp * expansion),
            ComplexConv2d(inp * expansion, inp * expansion, 3, stride, 1, bias=False, groups=inp * expansion),
            ComplexBatchNorm2d(inp * expansion),
            ComplexPReLU(inp * expansion),
            ComplexConv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            ComplexBatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ComplexConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ComplexConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = ComplexConv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = ComplexConv2d(inp, oup, k, s, p, bias=False)
        self.bn = ComplexBatchNorm2d(oup)
        if not linear:
            self.prelu = ComplexPReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

class ComplexMobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting):
        super(ComplexMobileFacenet, self).__init__()
        self.conv1 = ComplexConvBlock(2, 64, 3, 2, 1)  # Changed input channels from 3 to 2 (real and imaginary parts)
        self.dw_conv1 = ComplexConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = ComplexBottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ComplexConvBlock(self.inplanes, 512, 1, 1, 0)
        self.linear7 = ComplexConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = ComplexConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ComplexConv2d):
                if isinstance(m.kernel_size, int):
                    n = m.kernel_size * m.kernel_size * m.out_channels
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ComplexBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure that the input has the correct shape [batch_size, channels, height, width]
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x



class ComplexArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ComplexArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(2, 2, 112, 96))  # Changed input size to include real and imaginary parts
    net = ComplexMobileFacenet(bottleneck_setting=[(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1)])
    print(net)
    x = net(input)
    print(x.shape)
