# src/model.py
import torch
import torch.nn as nn


class GFactorPredictor(nn.Module):
    def __init__(self, input_size=9):
        super(GFactorPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128, bias=True)
        self.layer2 = nn.Linear(128, 64, bias=True)
        self.layer3 = nn.Linear(64, 16, bias=True)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x