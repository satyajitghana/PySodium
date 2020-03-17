'''
x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)
'''

import torch.nn as nn
import torch.nn.functional as F

from sodium.utils import setup_logger
from sodium.base import BaseModel

logger = setup_logger(__name__)


class QuizModel(BaseModel):
    def __init__(self, dropout_value=0.25):

        self.dropout_value = dropout_value

        super(QuizModel, self).__init__()

        # 32 x 32 x 3

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), dilation=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.convblock9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.convblock11 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        # 32 x 32 x 3

        x1 = self.convblock1(x)
        # 32 x 32 x 32
        x2 = self.convblock2(x1)
        # 32 x 32 x 32
        x3 = self.convblock3(x1 + x2)
        # 32 x 32 x 32
        x4 = self.pool1(x1 + x2 + x3)
        # 16 x 16 x 32

        x4 = self.convblock4(x4)
        # 16 x 16 x 64
        x5 = self.convblock5(x4)
        # 16 x 16 x 64
        x6 = self.convblock6(x4 + x5)
        # 16 x 16 x 64
        x7 = self.convblock6(x4 + x5 + x6)
        # 16 x 16 x 64
        x8 = self.pool2(x5 + x6 + x7)
        # 8 x 8 x 64

        x8 = self.convblock7(x8)
        # 8 x 8 x 128
        x9 = self.convblock8(x8)
        # 8 x 8 x 128
        x10 = self.convblock9(x8 + x9)
        # 8 x 8 x 128
        x11 = self.convblock10(x8 + x9 + x10)

        x12 = self.gap(x11)
        x13 = self.convblock11(x12)

        x13 = x13.view(-1, 10)

        return x13
