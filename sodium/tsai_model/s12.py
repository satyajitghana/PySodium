from sodium.model import ResNet
from sodium.model import BasicBlock


def TinyImageNetS12Model():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200)
