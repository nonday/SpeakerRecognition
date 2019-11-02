import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReLu20(nn.Module):
    """
     delta = min{ max{0,x},20}
    """
    def __init__(self):
        super(ReLu20, self).__init__()

    def forward(self, x):
        x = x.clamp(0,20)
        return x



def conv(in_planes, out_planes, k=3, stride=1, pad=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=k, stride=stride,
                     padding=pad, bias=False)

class ResBlock(nn.Module):
    
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()

        self.conv3x3_1 = conv(in_planes, out_planes, k=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv3x3_2 = conv(out_planes, out_planes, k=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu = ReLu20()


    def forward(self, x):
        residual = x

        out = self.conv3x3_1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3_2(out)
        out = self.bn2(out)

        out = out+residual
        out = self.relu(out)
        return out


class ResCNN(nn.Module):

    def __init__(self, inp=3, num_classes=1000):

        super(ResCNN, self).__init__()
        self.block = ResBlock
        self.conv = conv
        self.relu = ReLu20

        self.inp = inp
        self.layer1 = self._make_layer(self.inp, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)

        # average
        self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        # affine
        self.affine = nn.Linear(2048, 512)
        # ln
        self.ln = F.normalize

        # fc
        self.fc = nn.Linear(512, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, inp, oup):

        layers = []
        layers.append(self.conv(inp, oup, k=5, stride=2, pad=2))
        layers.append(nn.BatchNorm2d(oup))
        layers.append(self.relu())

        for i in range(3):
            layers.append(self.block(oup, oup, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.affine(x)

        x = self.ln(x)
        x = self.fc(x)
        return x


