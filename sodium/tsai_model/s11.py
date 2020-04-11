import torch
import torch.nn as nn
import torch.nn.functional as F

# C-B-R-C-B-R (X)


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        # C - B - R
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # C - B - (skip-connection) - R
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class LayerBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(LayerBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=planes,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=planes)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resblock = ResBlock(in_planes=planes, planes=planes)

    def forward(self, x):
        # C - M - B - R
        out = self.conv(x)
        out = self.mp(out)
        out = self.bn(out)
        out = F.relu(out)

        res = self.resblock(out)

        return out + res


class CIFAR10S11Model(nn.Module):
    def __init__(self):
        super(CIFAR10S11Model, self).__init__()
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.layer1 = LayerBlock(in_planes=64, planes=128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.layer3 = LayerBlock(in_planes=256, planes=512)

        self.mp = nn.MaxPool2d(kernel_size=4, stride=1)

        self.fc_layer = nn.Conv2d(
            in_channels=512, out_channels=10, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.mp(out)
        # out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        out = out.view(out.size(0), -1)

        return out
