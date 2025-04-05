import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from encoders.base_encoder import BaseEncoder


class ViTEncoder(BaseEncoder):
    """Vision Transformer encoder using torchvision model"""

    def __init__(self, img_size=224, in_channels=3):
        super().__init__()

        # Create the model without pretrained weights
        model = vit_b_16(weights=None)

        # If input channels are different, modify the patch embedding
        if in_channels != 3:
            model.conv_proj = nn.Conv2d(
                in_channels,
                model.conv_proj.out_channels,
                kernel_size=model.conv_proj.kernel_size,
                stride=model.conv_proj.stride,
            )

        # Keep only the features part, remove the classification head
        self.features = model.encoder
        self.patch_size = model.patch_size
        self.embed_dim = 768  # ViT-B has 768 dimension
        self.img_size = img_size

        # Store the embedding layer to transform patches
        self.patch_embed = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding

    def forward(self, x):
        # We need to adapt ViT's output to match the 2D spatial format expected by our interface
        B = x.shape[0]

        # Get patch embeddings
        x = self.patch_embed(x)
        patch_dim = x.shape[-1]

        # Reshape to sequence format for transformer
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embeddings and process through transformer
        x = x + self.pos_embedding
        x = self.features(x)

        # Remove class token and reshape back to 2D spatial format
        x = x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, patch_dim, patch_dim)

        return x

    def calc_feat_size(self, input_shape):
        # Override to handle the ViT's different output structure
        return self.embed_dim * (self.img_size // self.patch_size) ** 2
