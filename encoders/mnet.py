import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class MobileNetV3(BaseEncoder):
    """MobileNet‑V3‑Small backbone encoder (classifier removed)"""

    def __init__(self):
        super().__init__()

        # Build backbone without classifier
        backbone = mobilenet_v3_small(weights=None)

        # Modify the first convolutional layer for CIFAR-10 (32x32 images)
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels=first_conv.in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=3,    # smaller kernel
            stride=1,         # prevent aggressive downsampling
            padding=1,        # keep spatial size the same
            bias=first_conv.bias is not None
        )

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
