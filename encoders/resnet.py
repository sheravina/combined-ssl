from torchvision.models import resnet18, resnet50, resnet101
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class ResNetEncoder(BaseEncoder):
    """ResNet backbone encoder using torchvision models"""

    def __init__(self, model_type, zero_init_residual=False):
        super().__init__()

        # Create the model but without pretrained weights
        if model_type == ENC_RESNET18:
            model = resnet18(weights=None, zero_init_residual=zero_init_residual)
        elif model_type == ENC_RESNET50:
            model = resnet50(weights=None, zero_init_residual=zero_init_residual)
        elif model_type == ENC_RESNET101:
            model = resnet101(weights=None, zero_init_residual=zero_init_residual)
        elif model_type == ENC_RESNET50_PT:
            model = resnet50(weights='DEFAULT')
        elif model_type == ENC_RESNET18_PT:
            model = resnet18(weights='DEFAULT')            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Remove the classification head --  ends with the layer AdaptiveAvgPool2d
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Identity() # remove final fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x 
