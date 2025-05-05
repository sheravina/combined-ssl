from torch import nn
from encoders.base_encoder import BaseEncoder
from utils.constants import *
from .vision_transformers import vit_tiny, vit_small, vit_base

class ViTEncoder(BaseEncoder):
    """Vision Transformer backbone encoder"""

    def __init__(self, model_type, img_size=32, patch_size=4):
        super().__init__()
        
        # Import the vision transformer models from the source file
        
        
        # Create the model based on type
        if model_type == ENC_VIT_TINY:
            model = vit_tiny(patch_size=patch_size, img_size=[img_size])
        elif model_type == ENC_VIT_SMALL:
            model = vit_small(patch_size=patch_size, img_size=[img_size])
        elif model_type == ENC_VIT_BASE:
            model = vit_base(patch_size=patch_size, img_size=[img_size])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # The ViT model already returns the final embedding vector directly
        # so we don't need additional flattening layers
        self.features = model
        
    def forward(self, x):
        # The ViT forward function already returns the [CLS] token 
        # which serves as the image representation
        return self.features(x)