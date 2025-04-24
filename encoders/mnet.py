import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class MobileNetV3(BaseEncoder):
    """MobileNet‑V3‑Small backbone encoder (classifier removed)"""

    def __init__(self):
        super().__init__()

        # Build backbone without classifier
        backbone = mobilenet_v3_small(weights=None)   # or supply pretrained weights

        # Keep convolutional/SE blocks + global pooling
        self.features = nn.Sequential(
            backbone.features,   # conv/SE stack
            backbone.avgpool     # AdaptiveAvgPool2d(1)
        )
        self.flatten = nn.Flatten()  # → (batch, 576)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x                  # (N, 576)
