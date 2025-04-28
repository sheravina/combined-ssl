from timm import create_model
import torch.nn as nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *

class TinyViTEncoder(BaseEncoder):
    def __init__(self, model_type=ENC_TINYVIT, pretrained=True):
        super().__init__()

        if model_type == ENC_TINYVIT:
            self.model = create_model(
                'vit_small_patch16_224.augreg_in1k', 
                pretrained=pretrained,
                features_only=True,
                img_size=32,
                patch_size=4
            )
            
            # Modify the patch embedding layer to accept 32x32 inputs
            # self.model.patch_embed.img_size = (32, 32)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        features = self.model(x)
        x = features[-1]
        x = self.pool(x)
        x = self.flatten(x)
        return x