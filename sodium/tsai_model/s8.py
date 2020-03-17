import torch.nn as nn
import torch.nn.functional as F

from sodium.utils import setup_logger
from sodium.base import BaseModel

from sodium.model import ResNet18
from sodium.model import QuizModel


def CIFAR10S8Model():
    return ResNet18()


def QuizDNNModel():
    return QuizModel()
