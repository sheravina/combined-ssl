from torchvision.models import resnet18, resnet50
import torch.nn as nn
from encoders.base_encoder import BaseEncoder


class ResNetEncoder(BaseEncoder):
    """ResNet backbone encoder using torchvision models"""

    def __init__(self, model_type="resnet18", in_channels=3):
        super().__init__()

        # Create the model but without pretrained weights
        if model_type == "resnet18":
            model = resnet18(weights=None)
        elif model_type == "resnet50":
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # If input channels are different from the default 3, modify the first conv layer
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )

        # Remove the classification head
        self.features = nn.Sequential(*list(model.children())[:-2])

        # Store output channels for downstream tasks
        if model_type == "resnet18":
            self.out_channels = 512
        elif model_type == "resnet50":
            self.out_channels = 2048

    def forward(self, x):
        return self.features(x)
