# coding=utf-8
# author=yphacker


import torch.nn as nn
from conf import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=(1, 1)),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
                                   # nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
                                   nn.ReLU())
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
        #                            nn.ReLU())
        # self.linear1 = nn.Linear(2400, 64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24480, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, config.num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        # x = self.maxpool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x

# print(Model())
