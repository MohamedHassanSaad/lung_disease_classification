import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class ProposedCNN(nn.Module):
    def __init__(self, num_classes, dropout_rates=(0.5, 0.3, 0.3)):
        super(ProposedCNN, self).__init__()
        
        # Convolutional blocks
        self.block1 = ConvBlock(3, 32, dropout_rates[2])
        self.block2 = ConvBlock(32, 64, dropout_rates[2])
        self.block3 = ConvBlock(64, 128, dropout_rates[2])
        self.block4 = ConvBlock(128, 256, dropout_rates[2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, num_classes)
        
        # Dropout for fully connected layers
        self.dropout_fc1 = nn.Dropout(dropout_rates[1])
        self.dropout_fc2 = nn.Dropout(dropout_rates[1])
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.output(x)
        
        return x
