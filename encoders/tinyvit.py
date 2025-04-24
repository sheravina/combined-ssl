from timm import create_model
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class TinyViTEncoder(BaseEncoder):
    def __init__(self, model_type=ENC_TINYVIT, pretrained=False):
        super().__init__()

        if model_type == ENC_TINYVIT:
            self.model = create_model(
                'tiny_vit_21m_224.dist_in22k_ft_in1k', 
                pretrained=pretrained,
                features_only=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        features = self.model(x)
        x = features[-1]               # Use the last stage output
        x = self.pool(x)               # Global pooling
        x = self.flatten(x)            # Flatten to vector
        return x
