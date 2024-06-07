import torch
import torch.nn as nn

class Reconstruct(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3):
        super(Reconstruct, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.branch = nn.Conv2d(in_channels, out_channels, 1)
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        features = self.act(self.conv1(x))
        features = self.act(self.conv2(features)) + self.branch(x)
        return features
        