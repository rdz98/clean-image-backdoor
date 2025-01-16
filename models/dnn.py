import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 512)
        self.layer2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        output = self.layer1(x)
        output = self.layer2(output)
        return output
