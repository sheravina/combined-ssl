from torchvision.models import resnet18, resnet50
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class ResNetEncoder(BaseEncoder):
    """ResNet backbone encoder using torchvision models"""

    def __init__(self, model_type):
        super().__init__()

        # Create the model but without pretrained weights
        if model_type == ENC_RESNET18:
            model = resnet18(weights=None)
        elif model_type == ENC_RESNET50:
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Remove the classification head --  ends with the layer AdaptiveAvgPool2d
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x 
