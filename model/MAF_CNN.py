# -*- coding = utf-8 -*
# @Time：  10:07
# @File: MAF_CNN.py
# @Software: PyCharm
import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel=32, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y).view(b, c, 1)
        # print(y.shape)
        return x * y.expand_as(x)


class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 32, (5,), (1,), dilation=(2,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, (4,), (1,), dilation=(3,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, (3,), (1,), dilation=(4,)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.se = SELayer()

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x)
        # print(x2.shape)
        x3 = self.conv3(x)
        # print(x3.shape)
        out = torch.cat([x1, x2, x3], dim=2)
        # print(out.shape)
        out = self.se(out)
        # print(out.shape)
        return out


class MAF_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MAF_CNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, (256,), (32,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),

            nn.Dropout(),

            nn.Conv1d(64, 128, (8,), (1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, (8,), (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(1, 64, (256,), (32,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),

            nn.Dropout(),

            nn.Conv1d(64, 128, (5,), (1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, (5,), (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(1, 64, (256,), (32,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),

            nn.Dropout(),

            nn.Conv1d(64, 128, (7,), (1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, (7,), (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(1, 64, (256,), (32,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),

            nn.Dropout(),

            nn.Conv1d(64, 128, (6,), (1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, (6,), (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
        )
        self.dropout = nn.Dropout()
        self.msa = MSA()
        self.ft = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1216, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.out = nn.Linear(128, num_classes)

    def forward(self, x1, x2, x3, x4):
        x1 = self.cnn1(x1)
        x1 = self.msa(x1)
        x2 = self.cnn2(x2)
        x2 = self.msa(x2)
        x3 = self.cnn3(x3)
        x3 = self.msa(x3)
        x4 = self.cnn4(x4)
        x4 = self.msa(x4)
        x_concat = torch.cat([x1, x2, x3, x4], dim=2)
        # print(x_concat.shape)
        x = self.dropout(x_concat)
        x = self.ft(x)
        out = self.fc(x)
        x = self.out(out)
        return out, x
