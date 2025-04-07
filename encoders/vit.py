import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from encoders.base_encoder import BaseEncoder


class ViTEncoder(BaseEncoder):
    """Vision Transformer encoder using torchvision model"""

    def __init__(self, img_size=224, in_channels=3, output_dim=None):
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
        self.features = nn.Sequential(*model.encoder.layers)
        self.patch_size = model.patch_size
        self.embed_dim = 768  # ViT-B has 768 dimension
        self.img_size = img_size

        # Store the embedding layer to transform patches
        self.patch_embed = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        
        # Calculate expected sequence length based on image size
        self.seq_length = (img_size // model.patch_size) ** 2 + 1  # +1 for class token
        
        # Add projection to desired output dimension if specified
        self.output_dim = output_dim
        if output_dim is not None:
            self.projection = nn.Linear(self.embed_dim, output_dim)
        else:
            self.projection = None

    def forward(self, x):
        B = x.shape[0]
        
        # Get patch embeddings
        x = self.patch_embed(x)
        
        # Save spatial dimensions for reshaping later
        H = W = x.shape[-1]
        
        # Reshape to sequence format for transformer
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Check and interpolate position embeddings if needed
        if x.shape[1] != self.pos_embedding.shape[1]:
            pos_embed = self.pos_embedding
            cls_pos_embed = pos_embed[:, 0:1, :]
            other_pos_embed = pos_embed[:, 1:, :]
            
            expected_tokens = (H * W)
            orig_size = int(other_pos_embed.shape[1] ** 0.5)
            new_size = int(expected_tokens ** 0.5)
            
            if orig_size != new_size:
                other_pos_embed = other_pos_embed.reshape(-1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2)
                other_pos_embed = nn.functional.interpolate(
                    other_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
                other_pos_embed = other_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
                pos_embed = torch.cat((cls_pos_embed, other_pos_embed), dim=1)
            
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        
        # Process through transformer layers
        x = self.features(x)
        
        # Use only the class token for classification
        x = x[:, 0]  # Use CLS token only (B, embed_dim)
        
        # Apply projection if needed
        if self.projection is not None:
            x = self.projection(x)
        
        return x

    def calc_feat_size(self, input_shape):
        if self.output_dim is not None:
            return self.output_dim
        else:
            return self.embed_dim