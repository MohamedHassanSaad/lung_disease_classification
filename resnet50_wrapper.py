import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.5):
        super(ResNet50Wrapper, self).__init__()
        self.backbone = resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
