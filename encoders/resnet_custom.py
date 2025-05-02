import torch.nn as nn
import torchvision.models as models
from encoders.base_encoder import BaseEncoder

class CustomResNet(BaseEncoder):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load the ResNet architecture that will match the pretrained model
        resnet = models.resnet50(weights=None)
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Use the ResNet backbone as your feature extractor
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1]  # Remove the final classification layer
        )
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return x