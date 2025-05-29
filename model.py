import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DamageClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(DamageClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
